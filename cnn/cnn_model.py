import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers
from caveolae_cls.model import Model
from caveolae_cls.data_handler import DataHandler as DH
from caveolae_cls.cnn.cnn_data_handler import CNNDataHandler


class CNN(Model):

    def __init__(self, input_data_type, use_softmax=True):
        super(CNN, self).__init__(hp_fn="cnn/hyper_params.yaml")
        self.data_handler = CNNDataHandler(input_data_type, use_softmax=use_softmax)
        if input_data_type == "multiview" or input_data_type == "projection":
            self.input_shape = [self.hp['BATCH_SIZE'], DH.proj_dim, DH.proj_dim, 3]
        self.is_training = None
        self.num_classes = 2
        self.use_softmax = use_softmax

    def generate_input_placeholders(self):
        self.input_pl = tf.placeholder(tf.float32, shape=(self.input_shape))
        self.label_pl = tf.placeholder(tf.float32, shape=[self.hp['BATCH_SIZE'], self.num_classes] if self.use_softmax else self.hp['BATCH_SIZE'])
        self.is_training = tf.placeholder(tf.bool, shape=())

    def generate_model(self, input_pl=None, bn_decay=None, reuse=None):
        with tf.variable_scope(type(self).__name__):
            input_pl = self.input_pl if input_pl is None else input_pl

            input_channels = input_pl.get_shape()[-1].value
            batch_size = input_pl.get_shape()[0].value

            conv1 = nn_layers.conv2d(input_pl, input_channels, 32, (3, 3), 'conv1', is_training=self.is_training, reuse=reuse)
            pool1 = nn_layers.max_pool2d(conv1, (3, 3), 'pool1', reuse=reuse)
            conv2 = nn_layers.conv2d(pool1, 32, 64, (3, 3), 'conv2', is_training=self.is_training, reuse=reuse)
            pool2 = nn_layers.max_pool2d(conv2, (3, 3), 'pool2', reuse=reuse)
            conv3 = nn_layers.conv2d(pool2, 64, 128, (3, 3), 'conv3', is_training=self.is_training, reuse=reuse)
            pool3 = nn_layers.max_pool2d(conv3, (3, 3), 'pool3', reuse=reuse)
            conv4 = nn_layers.conv2d(pool3, 128, 256, (3, 3), 'conv4', is_training=self.is_training, reuse=reuse)
            pool4 = nn_layers.max_pool2d(conv4, (3, 3), 'pool4', reuse=reuse)
            conv5 = nn_layers.conv2d(pool4, 256, 512, (3, 3), 'conv5', is_training=self.is_training, reuse=reuse)
            pool5 = nn_layers.max_pool2d(conv5, (3, 3), 'pool5', reuse=reuse)
            # conv6 = nn_layers.conv2d(pool4, 256, 512, (1, 1), 'conv_6', is_training=self.is_training, reuse=reuse)
            # conv7 = nn_layers.conv2d(conv6, 512, 1024, (1, 1), 'conv_7', is_training=self.is_training, reuse=reuse)
            convnet = tf.reshape(pool5, [batch_size, -1])
            input_channels = convnet.get_shape()[-1].value
            fc1 = nn_layers.fc(convnet, input_channels, 256, 'fc1', is_training=self.is_training, reuse=reuse)
            fc2 = nn_layers.fc(fc1, 256, 128, 'fc2', is_training=self.is_training, reuse=reuse)
            self.features = fc2
            if self.use_softmax:
                 self.logits = nn_layers.fc(fc2, 128, self.num_classes, 'predicted_y', is_training=self.is_training, activation_fn=None,
                                            batch_norm=False, reuse=reuse)
                 self.pred = tf.nn.softmax(self.logits, name='softmax')
            else:
                self.pred = nn_layers.fc(fc2, 128, 1, 'predicted_y', is_training=self.is_training,
                                         activation_fn=tf.nn.sigmoid, batch_norm=False, reuse=reuse)
        return self.pred

    def generate_loss(self):
        """ pred: B*NUM_CLASSES,
            label: B, """
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_reg = 1
        for var in reg_variables:
            l2_reg += self.hp['BETA_REG'] * tf.nn.l2_loss(var)
        if self.use_softmax:
            logistic_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_pl,
                                                                      name='sigmoid_xentropy')
            self.loss = tf.reduce_mean(logistic_losses + l2_reg)
        else:
            simple_loss = -(self.label_pl * tf.log(self.pred + 1e-12) +
                     (1.0 - self.label_pl) * tf.log(1.0 - self.pred + 1e-12))
            cross_entropy = tf.reduce_sum(simple_loss, reduction_indices=[1])
            self.loss = tf.reduce_mean(cross_entropy + l2_reg)
        self.val_loss = self.loss

    def get_batch(self, use='train', val_set=None, cell_type=None):
        return self.data_handler.get_batch(self.input_shape, use=use, val_set=val_set, cell_type=cell_type)