import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.data_handler import DataHandler as DH
from caveolae_cls.cnn.cnn_data_handler import CNNDataHandler


class CNN(Model):

    def __init__(self, input_data_type, use_softmax=True):
        super(CNN, self).__init__(hp_fn="cnn/hyper_params.yaml")
        self.data_handler = CNNDataHandler(input_data_type, use_softmax)
        if input_data_type == "multiview" or input_data_type == "projection":
            self.input_shape = [self.hp['BATCH_SIZE'], DH.proj_dim, DH.proj_dim, 3]
        self.is_training = None
        self.use_softmax = use_softmax

    def generate_input_placeholders(self):
        self.input_pl = tf.placeholder(tf.float32, shape=(self.input_shape))
        self.label_pl = tf.placeholder(tf.float32, shape=[self.hp['BATCH_SIZE'], 2] if self.use_softmax else self.hp['BATCH_SIZE'])
        self.is_training = tf.placeholder(tf.bool, shape=())

    def generate_model(self, input_pl=None, bn_decay=None, reuse=None):
        input_pl = self.input_pl if input_pl is None else input_pl

        input_channels = input_pl.get_shape()[-1].value
        batch_size = input_pl.get_shape()[0].value

        pool0 = nn_layers.max_pool2d(input_pl, (2, 2), 'pool0', reuse=reuse)
        conv1 = nn_layers.conv2d(pool0, input_channels, 32, (3, 3), 'conv_1', is_training=self.is_training, reuse=reuse)
        conv2 = nn_layers.conv2d(conv1, 32, 64, (3, 3), 'conv_2', is_training=self.is_training, reuse=reuse)
        pool1 = nn_layers.max_pool2d(conv2, (3, 3), 'pool1', reuse=reuse)
        conv3 = nn_layers.conv2d(pool1, 64, 128, (3, 3), 'conv_3', is_training=self.is_training, reuse=reuse)
        pool2 = nn_layers.max_pool2d(conv3, (3, 3), 'pool2', reuse=reuse)
        conv4 = nn_layers.conv2d(pool2, 128, 256, (3, 3), 'conv_4', is_training=self.is_training, reuse=reuse)
        pool3 = nn_layers.max_pool2d(conv4, (3, 3), 'pool3', reuse=reuse)
        conv5 = nn_layers.conv2d(pool3, 256, 512, (3, 3), 'conv_5', is_training=self.is_training, reuse=reuse)
        pool4 = nn_layers.max_pool2d(conv5, (3, 3), 'pool4', reuse=reuse)
        # conv6 = nn_layers.conv2d(pool4, 256, 512, (1, 1), 'conv_6', is_training=self.is_training, reuse=reuse)
        # conv7 = nn_layers.conv2d(conv6, 512, 1024, (1, 1), 'conv_7', is_training=self.is_training, reuse=reuse)
        convnet = tf.reshape(pool4, [batch_size, -1])
        input_channels = convnet.get_shape()[-1].value
        fc1 = nn_layers.fc(convnet, input_channels, 256, 'fc1', is_training=self.is_training, reuse=reuse)
        fc2 = nn_layers.fc(fc1, 256, 128, 'fc2', is_training=self.is_training, reuse=reuse)
        if self.use_softmax:
             self.logits = nn_layers.fc(fc2, 128, 2, 'predicted_y', is_training=self.is_training, activation_fn=None,
                                        batch_norm=False, reuse=reuse)
             self.pred = tf.nn.softmax(self.logits, name='softmax')
        else:
            self.pred = nn_layers.fc(fc2, 128, 1, 'predicted_y', is_training=self.is_training,
                                     activation_fn=tf.nn.sigmoid, batch_norm=False, reuse=reuse)
        return self.pred

    def generate_loss(self):
        """ pred: B*NUM_CLASSES,
            label: B, """
        if self.use_softmax:
            logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_pl,
                                                                      name='sigmoid_xentropy')
            self.loss = tf.reduce_mean(logistic_losses)
        else:
            simple_loss = -(self.label_pl * tf.log(self.pred + 1e-12) +
                     (1.0 - self.label_pl) * tf.log(1.0 - self.pred + 1e-12))
            cross_entropy = tf.reduce_sum(simple_loss, reduction_indices=[1])
            self.loss = tf.reduce_mean(cross_entropy)

    def get_batch(self, eval=False, type='mixed'):
        return self.data_handler.get_batch(self.input_shape, eval=eval,
                                           type=type)