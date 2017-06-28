import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.cnn.cnn_model import CNN

class AutoEncoder(Model):

    proj_dim = int(CNN.proj_dim)

    def __init__(self, input_data_type):
        super(CNN, self).__init__(hp_fn="ae/hyper_params.yaml")
        self.data_handler = None # TODO
        if input_data_type == "multiview" or input_data_type == "projection":
            self.input_shape = [self.hp['BATCH_SIZE'], CNN.proj_dim, CNN.proj_dim, 3]
        self.is_training = None

    def get_batch(self, eval=False, type='mixed'):
        pass

    def generate_input_placeholders(self):
        self.input_pl = tf.placeholder(tf.float32, shape=(self.input_shape))
        self.output_pl = tf.placeholder(tf.float32, shape=(self.input_shape))
        self.is_training = tf.placeholder(tf.bool, shape=())

    def generate_model(self, input_pl=None, bn_decay=None, reuse=None):
        input_pl = self.input_pl if input_pl is None else input_pl

        input_channels = input_pl.get_shape()[-1].value
        # batch_size = input_pl.get_shape()[0].value

        # Encoder
        conv1 = nn_layers.conv2d(input_pl, input_channels, 32, (3, 3), 'conv1', is_training=self.is_training, reuse=reuse)
        pool1 = nn_layers.max_pool2d(conv1, (2, 2), 'pool1', reuse=reuse)
        conv2 = nn_layers.conv2d(pool1, 32, 64, (3, 3), 'conv3', is_training=self.is_training, reuse=reuse)
        pool2 = nn_layers.max_pool2d(conv2, (2, 2), 'pool2', reuse=reuse)
        conv3 = nn_layers.conv2d(pool2, 64, 128, (3, 3), 'conv4', is_training=self.is_training, reuse=reuse)
        pool3 = nn_layers.max_pool2d(conv3, (2, 2), 'pool3', reuse=reuse)
        conv4 = nn_layers.conv2d(pool3, 128, 256, (3, 3), 'conv5', is_training=self.is_training, reuse=reuse)
        pool4 = nn_layers.max_pool2d(conv4, (2, 2), 'pool4', reuse=reuse)

        # Decoder
        deconv1 = nn_layers.conv2d_transpose(pool4, 256, 128, (2,2), 'deconv1', is_training=self.is_training, reuse=reuse)
        deconv2 = nn_layers.conv2d_transpose(deconv1, 128, 64, (2, 2), 'deconv1', is_training=self.is_training,
                                             reuse=reuse)
        deconv3 = nn_layers.conv2d_transpose(deconv2, 64, 32, (2, 2), 'deconv1', is_training=self.is_training,
                                             reuse=reuse)
        self.pred = nn_layers.conv2d_transpose(deconv3, 32, input_channels, (2, 2), 'deconv1', is_training=self.is_training,
                                             reuse=reuse)
        return self.pred

    def generate_loss(self):
        self.loss = tf.reduce_mean(tf.squared_difference(self.pred, self.output_pl))