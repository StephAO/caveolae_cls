import tensorflow as tf
import numpy as np
from caveolae_cls.model import Model
import caveolae_cls.nn_layers as nn_layers
from itertools import izip


class SegmentedMIL(Model):

    def __init__(self, model):
        super(SegmentedMIL, self).__init__(hp_fn="segmented_mil/hyper_params.yaml")
        # self.hp = self.model.hp
        # self.model.hp['BATCH_SIZE'] = num_instance_per_bag
        self.model = model
        self.use_softmax = self.model.use_softmax
        self.data_handler = self.model.data_handler
        self.num_instances_per_bag = self.hp['NUM_INSTANCES']
        self.input_pl_shape = None
        self.is_training = None

    def generate_input_placeholders(self):
        self.input_pl_shape = [self.num_instances_per_bag] + self.model.input_shape[1:]
        print self.input_pl_shape
        self.input_pl = tf.placeholder(tf.float32, shape=self.input_pl_shape)
        self.label_pl = tf.placeholder(tf.float32, shape=[1, 2] if self.use_softmax else [1])
        self.model.generate_input_placeholders()
        self.is_training = self.model.is_training

    def generate_model(self, bn_decay=None):
        # i_preds = [None] * self.num_instances_per_bag
        self.model.generate_model(input_pl=self.input_pl, bn_decay=bn_decay, reuse=False)
        # i_preds[i] = tf.expand_dims(self.model.pred, 1)
        # instances = tf.concat(i_preds, 1)
        # print instances
        ##### Aggregation #####
        # self.pred = tf.reduce_mean(self.model.pred, axis=1)
        # self.features = tf.concat([tf.reduce_mean(instances, axis=1, keep_dims=True), tf.reduce_prod(instances, axis=1, keep_dims=True),
        #                            tf.reduce_max(instances, axis=1, keep_dims=True), tf.reduce_min(instances, axis=1, keep_dims=True),
        #                            tf.reduce_sum(instances, axis=1, keep_dims=True)], 1, name='mil_features')
        with tf.variable_scope(type(self).__name__):
            self.pred = tf.expand_dims(self.model.pred, axis=0)
            self.pred = nn_layers.noisy_and_1d(self.pred, 2 if self.use_softmax else 1)
        # self.features = tf.reshape(self.features, [self.hp['BATCH_SIZE'], -1])
        # self.logits = nn_layers.fc(self.features, 10, 2, 'predicted_y_mil', is_training=self.is_training, activation_fn=None)
        # self.pred = tf.nn.softmax(self.logits, name='softmax_mil')
        self.model.generate_model(bn_decay=bn_decay, reuse=True)
        return self.pred

    def generate_loss(self):
        if self.use_softmax:
            # loss_p = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=self.label_pl,
            #                                                  name='sigmoid_xentropy_mil')
            loss_l = tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.label_pl,
                                                             name='sigmoid_xentropy_mil')
            self.loss = tf.reduce_mean(loss_l)
        else:
            simple_loss = -(self.label_pl * tf.log(self.pred + 1e-12) +
                            (1.0 - self.label_pl) * tf.log(1.0 - self.pred + 1e-12))
            cross_entropy = tf.reduce_sum(simple_loss, reduction_indices=[1])
            self.loss = tf.reduce_mean(cross_entropy)
        self.val_loss = self.loss

        self.model.generate_loss()

    def get_batch(self, eval=False):
        # data = np.zeros(self.input_pl_shape)
        # labels = np.zeros([2] if self.use_softmax else ())
        for pos, neg in izip(self.data_handler.get_batch(self.input_pl_shape, eval=eval, type='positive'),
                             self.data_handler.get_batch(self.input_pl_shape, eval=eval, type='negative')):
            data = pos[0]
            labels = np.array([[0., 1.]]) if self.use_softmax else 1.
            yield data, labels

            data = neg[0]
            labels = np.array([[1., 0.]]) if self.use_softmax else 0.
            yield data, labels

    def save(self, sess, model_path, global_step=None):
        self.model.save(sess, model_path, global_step=global_step)
        super(SegmentedMIL, self).save(sess, model_path, global_step=global_step)

    def restore(self, sess, model_path):
        self.model.restore(sess, model_path)
        super(SegmentedMIL, self).restore(sess, model_path)
