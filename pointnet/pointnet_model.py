import tensorflow as tf
import numpy as np
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.pointnet.transform_nets import \
                                    input_transform_net, feature_transform_net
from caveolae_cls.pointnet.pointnet_data_handler import \
                                    PointNetDataHandler


class PointNet(Model):

    def __init__(self, use_softmax=True):
        super(PointNet, self).__init__(hp_fn="pointnet/hyper_params.yaml")
        self.data_handler = PointNetDataHandler(use_softmax=use_softmax)
        self.end_points = None
        self.reg_weight = 0.001
        self.output_shape = []
        self.is_training = None
        self.use_softmax = use_softmax

    def generate_input_placeholders(self):
        self.input_pl = tf.placeholder(tf.float32,  shape=(self.hp['BATCH_SIZE'], self.hp['NUM_POINTS'], 3))
        self.label_pl = tf.placeholder(tf.float32, shape=[self.hp['BATCH_SIZE'], 2] if self.use_softmax else self.hp['BATCH_SIZE'])
        self.is_training = tf.placeholder(tf.bool, shape=())

    def generate_global_features(self, input_pl=None, bn_decay=None, num_feats=4096):
        input_pl = self.input_pl if input_pl is None else input_pl

        batch_size = self.hp['BATCH_SIZE']
        num_point = self.hp['NUM_POINTS']
        end_points = {}

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(input_pl, self.is_training, bn_decay,
                                            K=3)
        point_cloud_transformed = tf.matmul(input_pl, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        input_channels = input_image.get_shape()[-1].value

        conv1 = nn_layers.conv2d(input_image, input_channels, 64, [1, 3],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=self.is_training,
                                 layer_name='conv1', batch_norm_decay=bn_decay)
        conv2 = nn_layers.conv2d(conv1, 64, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=self.is_training,
                                 layer_name='conv2', batch_norm_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(conv2, self.is_training, bn_decay,
                                              K=64)
        end_points['transform'] = transform
        self.end_points = end_points
        net_transformed = tf.matmul(tf.squeeze(conv2), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        input_channels = net_transformed.get_shape()[-1].value

        conv3 = nn_layers.conv2d(net_transformed, input_channels, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=self.is_training,
                                 layer_name='conv3', batch_norm_decay=bn_decay)
        conv4 = nn_layers.conv2d(conv3, 64, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=self.is_training,
                                 layer_name='conv4', batch_norm_decay=bn_decay)
        conv5 = nn_layers.conv2d(conv4, 256, 1024, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=self.is_training,
                                 layer_name='conv5', batch_norm_decay=bn_decay)
        conv6 = nn_layers.conv2d(conv5, 1024, num_feats, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=self.is_training,
                                 layer_name='conv6', batch_norm_decay=bn_decay)

        # Symmetric function: max pooling
        pool1 = nn_layers.max_pool2d(conv6, [num_point, 1],
                                     padding='VALID', layer_name='maxpool')

        self.global_feats = tf.reshape(pool1, [batch_size, -1])
        return self.global_feats

        # TODO
        # For mil, hypothesises on required changes:
        # 1. increase # features
        # 2. use conv after max pooling layer
        # Possible issue: locality of points might be lost

    def generate_model(self, input_pl=None, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx1 """
        gf = self.generate_global_features(input_pl=input_pl, bn_decay=bn_decay)

        input_channels = gf.get_shape()[-1].value

        fc1 = nn_layers.fc(gf, input_channels, 1024, batch_norm=True,
                           is_training=self.is_training, layer_name='fc1',
                           batch_norm_decay=bn_decay)
        # dp1 = nn_layers.dropout(fc1, keep_prob=0.7, is_training=is_training,
        #                         layer_name='dp1')
        fc2 = nn_layers.fc(fc1, 1024, 256, batch_norm=True, is_training=self.is_training,
                           layer_name='fc2', batch_norm_decay=bn_decay)
        # dp2 = nn_layers.dropout(fc2, keep_prob=0.7, is_training=is_training,
        #                         layer_name='dp2')
        fc3 = nn_layers.fc(fc2, 256, 64, batch_norm=True, is_training=self.is_training,
                           layer_name='fc3', batch_norm_decay=bn_decay)
        if self.use_softmax:
            self.logits = nn_layers.fc(fc3, 64, 2, 'predicted_y', is_training=self.is_training, activation_fn=None,
                                        batch_norm=False)
            self.pred = tf.nn.softmax(self.logits, name='softmax')
        else:
            self.pred = nn_layers.fc(fc3, 64, 1, 'predicted_y', is_training=self.is_training,
                                     activation_fn=tf.nn.sigmoid, batch_norm=False)

        return self.pred

    def generate_loss(self):
        """ pred: B*NUM_CLASSES,
            label: B, """
        if self.use_softmax:
            logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_pl,
                                                                      name='sigmoid_xentropy')
            classify_loss = tf.reduce_mean(logistic_losses)
        else:

            loss = -(self.label_pl * tf.log(self.pred + 1e-12) +
                     (1.0 - self.label_pl) * tf.log(1.0 - self.pred + 1e-12))
            cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
            classify_loss = tf.reduce_mean(cross_entropy)

        tf.summary.scalar('classify loss', classify_loss)

        # Enforce the transformation as orthogonal matrix
        transform = self.end_points['transform']  # BxKxK
        K = transform.get_shape()[1].value
        mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
        mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
        mat_diff_loss = tf.nn.l2_loss(mat_diff)
        tf.summary.scalar('mat loss', mat_diff_loss)

        self.loss = classify_loss + mat_diff_loss * self.reg_weight

    def get_batch(self, eval=False, type='mixed'):
        return self.data_handler.get_batch([self.hp['BATCH_SIZE'],
                                            self.hp['NUM_POINTS'], 3],
                                           eval=eval, type=type)
