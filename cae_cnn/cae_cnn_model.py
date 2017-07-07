import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers
import caveolae_cls.cae.cae_model as cae
import caveolae_cls.cnn.cnn_model as cnn
from caveolae_cls.cnn.cnn_data_handler import CNNDataHandler
from caveolae_cls.data_handler import DataHandler as DH
from caveolae_cls.model import Model


class CAE_CNN(Model):

    def __init__(self, input_data_type, use_softmax=True):
        super(CAE_CNN, self).__init__(hp_fn="cnn/hyper_params.yaml")
        self.data_handler = CNNDataHandler(input_data_type, use_softmax)
        if input_data_type == "multiview" or input_data_type == "projection":
            self.input_shape = [self.hp['BATCH_SIZE'], DH.proj_dim, DH.proj_dim, 3]
        self.is_training = None
        self.use_softmax = use_softmax
        self.cae = cae.CAE(input_data_type)
        self.cnn = cnn.CNN(input_data_type, use_softmax=use_softmax)

    def generate_input_placeholders(self):
        self.input_pl = tf.placeholder(tf.float32, shape=(self.input_shape))
        self.label_pl = tf.placeholder(tf.float32, shape=[self.hp['BATCH_SIZE'], 2] if self.use_softmax else self.hp['BATCH_SIZE'])
        self.is_training = tf.placeholder(tf.bool, shape=())

    def generate_model(self, input_pl=None, bn_decay=None, reuse=None):
        input_pl = self.input_pl if input_pl is None else input_pl
        in_channels = input_pl.get_shape()[-1].value
        self.features = self.cae.encode(input_pl, in_channels)
        self.replicator = self.cae.decode(in_channels)
        self.classifier = self.cnn.generate_model(input_pl=self.features, bn_decay=bn_decay, reuse=reuse)

    def generate_loss(self):
        """ pred: B*NUM_CLASSES,
            label: B, """
        self.cae.generate_loss()
        self.cnn.generate_loss()
        self.loss = self.cae.loss + self.cnn.loss
        self.val_loss = self.cnn.loss

    def get_batch(self, eval=False, type='mixed'):
        return self.data_handler.get_batch(self.input_shape, eval=eval,
                                           type=type)