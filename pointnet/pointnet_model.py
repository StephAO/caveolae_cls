import tensorflow as tf
import numpy as np
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.pointnet.transform_nets import \
                                    input_transform_net, feature_transform_net
from caveolae_cls.pointnet.pointnet_data_handler import \
                                    PointNetDataHandler


class PointNet(Model):

    def __init__(self):
        super(PointNet, self).__init__(hp_fn="pointnet/hyper_params.yaml")
        self.data_handler = PointNetDataHandler()
        self.end_points = None
        self.reg_weight = 0.001
        self.output_shape = []

    def get_input_placeholders(self, batch_size):
        pointclouds_pl = tf.placeholder(tf.float32,
                                        shape=(batch_size,
                                               self.hp['NUM_POINTS'], 3))
        labels_pl = tf.placeholder(tf.float32, shape=batch_size)
        return pointclouds_pl, labels_pl

    def get_model(self, point_cloud, is_training, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        end_points = {}

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(point_cloud, is_training, bn_decay,
                                            K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        input_channels = input_image.get_shape()[-1].value

        conv1 = nn_layers.conv2d(input_image, input_channels, 64, [1, 3],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv1', batch_norm_decay=bn_decay)
        conv2 = nn_layers.conv2d(conv1, 64, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv2', batch_norm_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(conv2, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        self.end_points = end_points
        net_transformed = tf.matmul(tf.squeeze(conv2), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        input_channels = net_transformed.get_shape()[-1].value

        conv3 = nn_layers.conv2d(net_transformed, input_channels, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv3', batch_norm_decay=bn_decay)
        conv4 = nn_layers.conv2d(conv3, 64, 128, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv4', batch_norm_decay=bn_decay)
        conv5 = nn_layers.conv2d(conv4, 128, 1024, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv5', batch_norm_decay=bn_decay)

        # Symmetric function: max pooling
        pool1 = nn_layers.max_pool2d(conv5, [num_point, 1],
                                     padding='VALID', layer_name='maxpool')

        lnet = tf.reshape(pool1, [batch_size, -1])

        input_channels = lnet.get_shape()[-1].value

        fc1 = nn_layers.fc(lnet, input_channels, 512, batch_norm=True,
                           is_training=is_training, layer_name='fc1',
                           batch_norm_decay=bn_decay)
        dp1 = nn_layers.dropout(fc1, keep_prob=0.7, is_training=is_training,
                                layer_name='dp1')
        fc2 = nn_layers.fc(dp1, 512, 256, batch_norm=True, is_training=is_training,
                           layer_name='fc2', batch_norm_decay=bn_decay)
        dp2 = nn_layers.dropout(fc2, keep_prob=0.7, is_training=is_training,
                                layer_name='dp2')
        pred = nn_layers.fc(dp2, 256, 1, activation_fn=tf.nn.sigmoid,
                            layer_name='fc3', is_training=is_training)

        return pred

    def get_loss(self, pred, label):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
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

        return classify_loss + mat_diff_loss * self.reg_weight
