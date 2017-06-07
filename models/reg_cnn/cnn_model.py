import tensorflow as tf
import numpy as np
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.models.regcnn.pointnet_data_handler import \
                                    PointNetDataHandler


class PointNet(Model):

    def __init__(self):
        super(PointNet, self).__init__(hp_fn="models/pointnet/hyper_params.yaml")
        self.data_handler = PointNetDataHandler()

    def get_input_placeholders(self, batch_size):
        input_pl = tf.placeholder(tf.float32,
                                        shape=(batch_size,
                                               self.hp['NUM_POINTS'], 3))
        labels_pl = tf.placeholder(tf.float32, shape=batch_size)
        return input_pl, labels_pl

    def get_model(self, input_pl, is_training, bn_decay=None):

        input_channels = input_pl.get_shape()[-1].value

        conv1 = nn_layers.conv2d(input_pl, input_channels, 32, (3, 3), 'conv_1')
        conv2 = nn_layers.conv2d(conv1, 32, 64, (3, 3), 'conv_2')
        pool1 = nn_layers.max_pool2d(conv2, (3, 3), 'pool1')
        conv3 = nn_layers.conv2d(pool1, 64, 128, (3, 3), 'conv_3')
        pool2 = nn_layers.max_pool2d(conv3, (3, 3), 'pool2')
        conv4 = nn_layers.conv2d(pool2, 128, 128, (3, 3), 'conv_4')
        pool3 = nn_layers.max_pool2d(conv4, (3, 3), 'pool3')
        conv5 = nn_layers.conv2d(pool3, 128, 256, (3, 3), 'conv_5')
        pool4 = nn_layers.max_pool2d(conv5, (3, 3), 'pool4')
        conv6 = nn_layers.conv2d(pool4, 256, 1024, (1, 1), 'conv_6')
        conv7 = nn_layers.conv2d(conv6, 1024, 2, (1, 1), 'conv_7')
        n_and = nn_layers.noisy_and(conv7, 2, 'noisy_and')
        y_hat = nn_layers.fc_layer(n_and, 2, 1, 'predicted_y', act=tf.nn.sigmoid,
                                   use_batch_norm=False)

        diff = input_labels * tf.log(tf.clip_by_value(y_hat, 1e-16, 1.0))
        cross_entropy = -tf.reduce_mean(diff)

        return net, end_points

    def get_loss(self, pred, label, **reg_kwargs):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        classify_loss = tf.reduce_mean(cross_entropy)

        return classify_loss
