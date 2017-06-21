import tensorflow as tf
import numpy as np
from caveolae_cls.model import Model
import caveolae_cls.nn_layers as nn_layers
from itertools import izip


class SegmentedMIL(Model):

    def __init__(self, model, num_instance_per_bag=10):
        super(SegmentedMIL, self).__init__(hp_fn="segmented_mil/hyper_params.yaml")
        self.model = model
        self.hp = self.model.hp
        # self.model.hp['BATCH_SIZE'] = num_instance_per_bag
        self.num_instances_per_bag = num_instance_per_bag
        self.input_pl_shape = None
        self.is_training = None

    def generate_input_placeholders(self):
        self.input_pl_shape = [self.hp['BATCH_SIZE']] + [self.num_instances_per_bag] + self.model.input_shape[1:]
        self.input_pl = tf.placeholder(tf.float32, shape=self.input_pl_shape)
        self.label_pl = tf.placeholder(tf.float32, shape=[self.hp['BATCH_SIZE']])
        self.model.generate_input_placeholders()
        self.is_training = self.model.is_training

    def generate_model(self, bn_decay=None):
        i_preds = [None] * self.num_instances_per_bag
        for i in xrange(self.num_instances_per_bag):
            reuse = True if i > 0 else None
            instance = self.model.generate_model(input_pl=self.input_pl[:, i, :, :, :], bn_decay=bn_decay, reuse=reuse)
            i_preds[i] = tf.expand_dims(instance, 1)
        instances = tf.concat(i_preds, 1)
        # Aggregation
        # self.pred = tf.reduce_mean(instances, axis=1)
        self.pred = nn_layers.noisy_and_1d(instances, 2 if self.model.use_softmax else 1)
        self.model.generate_model(bn_decay=bn_decay, reuse=True)
        return self.pred

    def generate_loss(self):
        loss = -(self.label_pl * tf.log(self.pred + 1e-12) +
                 (1.0 - self.label_pl) * tf.log(1.0 - self.pred + 1e-12))
        cross_entropy = tf.reduce_sum(loss, reduction_indices=[1])
        self.loss = tf.reduce_mean(cross_entropy)
        self.model.generate_loss()

    def get_batch(self, eval=False):
        data = np.zeros(self.input_pl_shape)
        labels = np.zeros(self.hp['BATCH_SIZE'])
        i = 0
        for pos, neg in izip(self.model.data_handler.get_batch(self.input_pl_shape[1:], eval=eval, type='positive'),
                             self.model.data_handler.get_batch(self.input_pl_shape[1:], eval=eval, type='negative')):
            data[i] = pos[0]
            labels[i] = 1
            data[i + 1] = neg[0]
            labels[i + 1] = 0
            i += 2
            if i >= self.hp['BATCH_SIZE']:
                yield data, labels
                i = 0
