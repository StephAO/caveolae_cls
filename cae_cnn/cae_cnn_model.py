import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers
import caveolae_cls.cae.cae_model as cae
import caveolae_cls.cnn.cnn_model as cnn
from caveolae_cls.cnn.cnn_data_handler import CNNDataHandler
from caveolae_cls.data_handler import DataHandler as DH
from caveolae_cls.model import Model


class CAE_CNN(Model):

    def __init__(self, input_data_type, use_softmax=True):
        super(CNN, self).__init__(hp_fn="cnn/hyper_params.yaml")
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
        self.features = self.cae.encode()

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