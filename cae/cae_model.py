import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter

import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.data_handler import DataHandler as DH
from caveolae_cls.cae.cae_data_handler import CAEDataHandler

class CAE(Model):

    def __init__(self, input_data_type):
        super(CAE, self).__init__(hp_fn="cae/hyper_params.yaml")
        self.data_handler = CAEDataHandler(input_data_type)
        if input_data_type == "multiview" or input_data_type == "projection":
            self.input_shape = [self.hp['BATCH_SIZE'], DH.proj_dim, DH.proj_dim, 3]
            self.feature_shape = DH.feature_shape
        self.is_training = None
        self.use_softmax = False
        self.gauss_var = 10
        self.pred = None
        self.gaussian_kernel = None

    def get_batch(self, eval=False, type='mixed'):
        return self.data_handler.get_batch(self.input_shape, eval=eval, type=type)

    def generate_input_placeholders(self):
        self.input_pl = tf.placeholder(tf.float32, shape=(self.input_shape))
        self.is_training = tf.placeholder(tf.bool, shape=())
        
    def encode(self, input_pl, in_channels):
        # Encoder
        # pool0 = nn_layers.max_pool2d(input_pl, (2,2), 'pool0')
        conv1 = nn_layers.conv2d(input_pl, in_channels, 32, (3, 3), 'conv1', is_training=self.is_training)
        pool1 = nn_layers.max_pool2d(conv1, (2, 2), 'pool1')
        conv2 = nn_layers.conv2d(pool1, 32, 64, (3, 3), 'conv2', is_training=self.is_training)
        pool2 = nn_layers.max_pool2d(conv2, (2, 2), 'pool2')
        conv3 = nn_layers.conv2d(pool2, 64, 128, (3, 3), 'conv3', is_training=self.is_training)
        pool3 = nn_layers.max_pool2d(conv3, (2, 2), 'pool3')
        conv4 = nn_layers.conv2d(pool3, 128, 256, (3, 3), 'conv4', is_training=self.is_training)
        pool4 = nn_layers.max_pool2d(conv4, (2, 2), 'pool4')
        conv5 = nn_layers.conv2d(pool4, 256, self.feature_shape[-1], (3, 3), 'conv5', is_training=self.is_training)

        self.features = conv5 # nn_layers.max_pool2d(conv5, (2, 2), 'pool5')

    def decode(self, in_channels):
        # Decoder
        deconv1 = nn_layers.conv2d_transpose(self.features, self.feature_shape[-1], 256, (3, 3), 'deconv1',
                                             stride=[1, 1], is_training=self.is_training)
        deconv2 = nn_layers.conv2d_transpose(deconv1, 256, 128, (3, 3), 'deconv2', is_training=self.is_training)
        deconv3 = nn_layers.conv2d_transpose(deconv2, 128, 64, (3, 3), 'deconv3', is_training=self.is_training)
        deconv4 = nn_layers.conv2d_transpose(deconv3, 64, 32, (3, 3), 'deconv4', is_training=self.is_training)
        deconv5 = nn_layers.conv2d_transpose(deconv4, 32, in_channels, (3, 3), 'deconv5', is_training=self.is_training,
                                             activation_fn=tf.nn.sigmoid)
        self.pred = deconv5

        if self.pred.get_shape().as_list() != self.input_shape:
            print "Predicted shape:", str(self.pred.get_shape().as_list())
            print "Input shape:", str(self.input_shape)
            raise TypeError("Output shape not the same as input shape")

    def generate_model(self, input_pl=None, bn_decay=None, reuse=None):
        input_pl = self.input_pl if input_pl is None else input_pl
        in_channels = input_pl.get_shape()[-1].value
        self.encode(input_pl, in_channels)
        self.decode(in_channels)

        return self.pred

    def generate_gaussian_kernel(self):
        gaussian = np.zeros([9, 9])
        gaussian[4][4] = 1
        gaussian = gaussian_filter(gaussian, 1)
        gaussian_kernel = np.zeros([9, 9, 1, 1])
        gaussian_kernel[:, :, 0, 0] = gaussian
        self.gaussian_kernel = tf.constant(gaussian_kernel, dtype=tf.float32)

    def euclidean_loss(self):
        self.generate_gaussian_kernel()
        loss = 0.
        for i in xrange(self.pred.get_shape().as_list()[-1]):
            pred_input = self.pred[:, :, :, i:i + 1]
            real_input = self.input_pl[:, :, :, i:i + 1]
            # loss += tf.reduce_sum(tf.abs(pred_input - 10000. * real_input))
            pred_gauss = tf.nn.conv2d(pred_input, self.gaussian_kernel, [1, 1, 1, 1], "SAME")
            real_gauss = tf.nn.conv2d(real_input, self.gaussian_kernel, [1, 1, 1, 1], "SAME")
            loss += tf.reduce_sum(tf.abs(pred_gauss - real_gauss))
        return loss

    def jaccard_index(self):
        intersection = tf.reduce_sum(self.pred * self.input_pl)
        union = tf.reduce_sum(self.pred) + tf.reduce_sum(self.input_pl) - intersection
        iou = intersection / union
        return iou

    def dice_index(self):
        intersection = tf.reduce_sum(self.pred * self.input_pl)
        union = tf.reduce_sum(self.pred) + tf.reduce_sum(self.input_pl)
        iou = intersection / union
        return iou

    def generate_loss(self):

        self.loss = 1 / self.dice_index() + (tf.reduce_sum(self.pred) - tf.reduce_sum(self.input_pl))
        self.val_loss = self.euclidean_loss()

    # def generate_loss(self):
    #     loss = 0
    #     for batch in xrange(self.input_shape[0]):
    #         for channel in xrange(self.input_shape[-1]):
    #             loss += tf.reduce_sum(tf.abs(self.pred[batch, :, :, channel] -
    #                                          self.input_pl[batch, :, :, channel]))
    #     # loss2 = gaussian_filter(self.pred, 2 * self.gauss_var) - gaussian_filter(self.input_pl, 2 * self.gauss_var)
    #     self.loss = loss
        # self.loss = tf.reduce_sum(tf.abs(self.pred - self.input_pl))

