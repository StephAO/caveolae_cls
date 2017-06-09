import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.cnn.cnn_data_handler import \
                                    CNNDataHandler


class CNN(Model):

    def __init__(self, input_data_type):
        super(CNN, self).__init__(hp_fn="pointnet/hyper_params.yaml")
        self.data_handler = CNNDataHandler(input_data_type)
        if input_data_type == "multiview" or input_data_type == "projections":
            self.input_shape = [None, 600, 600, 3]
        elif input_data_type == "voxels":
            self.input_shape = [None, 600, 600, 600]

    def get_input_placeholders(self, batch_size):
        self.input_shape[0] = batch_size
        input_pl = tf.placeholder(tf.float32,
                                        shape=(self.input_shape))
        labels_pl = tf.placeholder(tf.float32, shape=batch_size)
        return input_pl, labels_pl

    def get_model(self, input_pl, is_training, bn_decay=None):

        input_channels = input_pl.get_shape()[-1].value
        batch_size = input_pl.get_shape()[0].value

        conv1 = nn_layers.conv2d(input_pl, input_channels, 32, (3, 3), 'conv_1', is_training=is_training)
        conv2 = nn_layers.conv2d(conv1, 32, 64, (3, 3), 'conv_2', is_training=is_training)
        pool1 = nn_layers.max_pool2d(conv2, (3, 3), 'pool1')
        conv3 = nn_layers.conv2d(pool1, 64, 128, (3, 3), 'conv_3', is_training=is_training)
        # pool2 = nn_layers.max_pool2d(conv3, (3, 3), 'pool2')
        # conv4 = nn_layers.conv2d(pool2, 128, 128, (3, 3), 'conv_4', is_training=is_training)
        # pool3 = nn_layers.max_pool2d(conv4, (3, 3), 'pool3')
        # conv5 = nn_layers.conv2d(pool3, 128, 256, (3, 3), 'conv_5', is_training=is_training)
        # pool4 = nn_layers.max_pool2d(conv5, (3, 3), 'pool4')
        # conv6 = nn_layers.conv2d(pool4, 256, 512, (1, 1), 'conv_6', is_training=is_training)
        # conv7 = nn_layers.conv2d(conv6, 512, 1024, (1, 1), 'conv_7', is_training=is_training)
        convnet = tf.reshape(conv3, [batch_size, -1])
        input_channels = convnet.get_shape()[-1].value
        fc1 = nn_layers.fc(convnet, input_channels, 512, 'fc1', is_training=is_training)
        fc2 = nn_layers.fc(fc1, 512, 256, 'fc2', is_training=is_training)
        pred = nn_layers.fc(fc2, 256, 1, 'predicted_y', activation_fn=tf.nn.sigmoid, batch_norm=False)

        return pred

    def get_loss(self, pred, label):
        """ pred: B*NUM_CLASSES,
            label: B, """
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        classify_loss = tf.reduce_mean(cross_entropy)

        return classify_loss
