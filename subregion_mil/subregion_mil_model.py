import tensorflow as tf
from caveolae_cls.model import Model


class SubregionMIL(Model):

    def __init__(self, model):
        self.model = model

    def __init__(self, model, input_data_type):
        self.model = model
        self.input_data_type = input_data_type

    def get_input_placeholders(self, batch_size):
        return self.model.get_input_placeholders(batch_size)

    def get_model(self, input_pl, is_training, bn_decay=None):
        preds = [None] * self.num_instances_per_bag
        for i in xrange(self.num_instances_per_bag):
            preds[i] = self.model.get_model(input_pl[i], is_training,
                                            bn_decay=bn_decay)
        instances = tf.concat(preds, 1)
        # Aggregate using some kind of pooling function (noisy-and, max...)
        #

    def get_loss(self, pred, label):
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        classify_loss = tf.reduce_mean(cross_entropy)

        return classify_loss