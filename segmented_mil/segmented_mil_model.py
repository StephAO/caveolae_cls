import tensorflow as tf
import numpy as np
from caveolae_cls.model import Model
import caveolae_cls.nn_layers as nn_layers
from itertools import izip


class SegmentedMIL(Model):

    def __init__(self, model, num_instance_per_bag=5):
        self.model = model
        self.hp = self.model.hp
        self.model.hp['BATCH_SIZE'] = num_instance_per_bag
        self.model.input_shape[0] = num_instance_per_bag
        self.num_instances_per_bag = num_instance_per_bag

    def get_input_placeholders(self):
        input_pl_shape = [self.num_instances_per_bag] + self.model.input_shape
        input_pl = np.zeros(input_pl_shape)
        for i in xrange(self.num_instances_per_bag):
              input_pl[i], _ = \
                  self.model.get_input_placeholders()
        labels_pl = tf.placeholder(tf.float32, shape=1)
        return input_pl, labels_pl

    def get_model(self, input_pl, is_training, bn_decay=None):
        preds = [None] * self.num_instances_per_bag
        for i in xrange(self.num_instances_per_bag):
            reuse = True if i > 0 else None
            pred = self.model.get_model(input_pl[i], is_training,
                                        bn_decay=bn_decay, reuse=reuse)
            preds[i] = tf.expand_dims(pred, 1)
        instances = tf.concat(preds, 1)
        # Aggregation
        bag = nn_layers.noisy_and_1d(instances, 1)
        return bag

    def get_loss(self, pred, label):
        print "------", np.shape(pred), np.shape(label)
        print "******", pred.get_shape(), label.get_shape()
        loss = -(label * tf.log(pred + 1e-12) +
                 (1.0 - label) * tf.log(1.0 - pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        classify_loss = tf.reduce_mean(cross_entropy)

        return classify_loss

    def get_batch(self, eval=False):
        for pos, neg in izip(self.model.get_batch(eval=eval, type='positive'),
                             self.model.get_batch(eval=eval, type='negative')):
            yield pos[0], 1
            yield neg[0], 0
