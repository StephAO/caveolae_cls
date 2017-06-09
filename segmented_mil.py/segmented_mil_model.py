import tensorflow as tf
from caveolae_cls.model import Model
import caveolae_cls.nn_layers as nn_layers


class SegmentedMIL(Model):

    def __init__(self, model):
        self.model = model

    def __init__(self, model, input_data_type, num_instance_per_bag=5):
        self.model = model
        self.input_data_type = input_data_type
        self.num_instances_per_bag = num_instance_per_bag

    def get_input_placeholders(self, batch_size):
        input_pl = [None] * self.num_instances_per_bag
        labels_pl = [None] * self.num_instances_per_bag
        for i in xrange(self.num_instances_per_bag):
              input_pl[i], labels_pl[i] = \
                  self.model.get_input_placeholders(batch_size)
        return input_pl, labels_pl

    def get_model(self, input_pl, is_training, bn_decay=None):
        preds = [None] * self.num_instances_per_bag
        for i in xrange(self.num_instances_per_bag):
            pred = self.model.get_model(input_pl[i], is_training,
                                            bn_decay=bn_decay)
            preds[i] = tf.expand_dims(pred, 1)
        instances = tf.concat(preds, 1)
        # Aggregate using some kind of pooling function (noisy-and, max...)
        bag = nn_layers.noisy_and_1d(instances, 1)
        return bag

    def get_loss(self, pred, label):
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        classify_loss = tf.reduce_mean(cross_entropy)

        return classify_loss