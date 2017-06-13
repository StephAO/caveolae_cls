import tensorflow as tf
from caveolae_cls.model import Model
import caveolae_cls.nn_layers as nn_layers


class SubregionMIL(Model):

    def __init__(self, model):
        self.model = model

    def __init__(self, model, input_data_type):
        self.model = model
        self.input_data_type = input_data_type

    def get_input_placeholders(self):
        return self.model.get_input_placeholders()

    def get_model(self, input_pl, is_training, bn_decay=None):
        num_feats = 1024
        gf = self.model.get_global_features(input_pl, is_training,
                                       bn_decay=bn_decay, num_feats=num_feats)
        conv4 = nn_layers.conv2d(gf, num_feats, 128, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv4', batch_norm_decay=bn_decay)
        conv5 = nn_layers.conv2d(conv4, 128, 1024, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 batch_norm=True, is_training=is_training,
                                 layer_name='conv5', batch_norm_decay=bn_decay)

        # Aggregate using some kind of pooling function (noisy-and, max...)
        nn_layers.noisy_and(instances,)

    def get_loss(self, pred, label):
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        classify_loss = tf.reduce_mean(cross_entropy)

        return classify_loss