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
        self.input_pl = None
        self.label_pl = None

    def generate_input_placeholders(self):
        self.input_pl_shape = [self.num_instances_per_bag] + self.model.input_shape[1:]
        print self.input_pl_shape
        self.input_pl = tf.placeholder(tf.float32, shape=self.input_pl_shape)
        self.label_pl = tf.placeholder(tf.float32, shape=[1, 2] if self.use_softmax else [1])
        self.model.generate_input_placeholders()
        self.is_training = self.model.is_training

    def choose_2_centroids(self):
        # Step 0: Initialisation: Select `n_clusters` number of random points
        centroid_0 = self.input_pl[0]
        distances_to_others = tf.reduce_sum(tf.square(tf.subtract(self.input_pl[1:], self.input_pl[0])), axis=(1, 2, 3))
        furthest_index = tf.argmax(distances_to_others)
        centroid_1 = self.input_pl[furthest_index]
        return tf.concat([centroid_0, centroid_1], 0)

    def assign_to_nearest(self, centroids):
        # Finds the nearest centroid for each sample

        # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
        expanded_vectors = tf.expand_dims(self.input_pl, 0)
        expanded_centroids = tf.expand_dims(centroids, 1)
        distances = tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2)
        mins = tf.argmin(distances, 0)
        # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
        nearest_indices = mins
        return nearest_indices

    def update_centroids(self, nearest_indices):
        # Updates the centroid to be the mean of all samples associated with it.
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(self.input_pl, nearest_indices, 2)
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
        return new_centroids

    def cluster(self, num_iterations=10):
        centroids = self.choose_2_centroids()
        nearest_indices = self.assign_to_nearest(centroids)
        for i in xrange(num_iterations):
            centroids = self.update_centroids(nearest_indices)
            nearest_indices = self.assign_to_nearest(centroids)
        # since index will either be 0 or 1, if more than half of indices are 1, then the rounded mean will be 1
        smaller_cluster = tf.rint(tf.reduce_mean(nearest_indices))
        in_smaller_cluster = tf.equal(nearest_indices, smaller_cluster)
        input_idx_in_cluster = tf.where(in_smaller_cluster)
        return input_idx_in_cluster

    def get_positive_mean(self, values, num_iterations=10):
        """
        # TODO Make better. Either remove or better determine num_iterations
        Uses a simplified k-means (2-means) to divide positive and negative instances.
        Positive and negative refer to instance level classification (i.e. positives are closer to 1, negatives closer to 0)
        Initializes negative centroid to min of data, Initializes positive centroid to max of data
        Iteratively:
            1. Find which centroid each value is closest to.
            2. Updates centroid means based on grouped values.
        Return indices of values that are closer to larger centroid (positives).

        Args:
            data[List of floats]: Represents probability of being positive (1=definitely positive, 0=definitely negative)
            num_iterations[Int]: Number of times to update centroids
        Returns:
            Tensor of indices representing

        K-Means Clustering using TensorFlow.
        'vectors' should be a n*k 2-D NumPy array, where n is the number
        of vectors of dimensionality k.
        """
        centroid_neg = tf.reduce_min(values)
        centroid_pos = tf.reduce_max(values)

        for i in xrange(num_iterations):
            mid_point = (centroid_pos + centroid_neg) / 2
            distances = values - mid_point
            centroid_neg_distances = tf.clip_by_value(distances, -1., 0.)
            centroid_pos_distances = tf.clip_by_value(distances, 0., 1.)
            centroid_neg_new_mean = tf.reduce_sum(centroid_neg_distances) / (
            tf.cast(tf.count_nonzero(centroid_neg_distances), tf.float32) + 1e-20)
            centroid_pos_new_mean = tf.reduce_sum(centroid_pos_distances) / (
            tf.cast(tf.count_nonzero(centroid_pos_distances), tf.float32) + 1e-20)
            centroid_neg = centroid_neg_new_mean + mid_point
            centroid_pos = centroid_pos_new_mean + mid_point

        mid_point = (centroid_pos + centroid_neg) / 2
        distances = values - mid_point
        centroid_pos_distances = tf.clip_by_value(distances, 0., 1.)
        zero = tf.constant(0, dtype=tf.float32)
        non_zeros = tf.not_equal(centroid_pos_distances, zero)
        pos_indices = tf.where(non_zeros)
        positives = tf.gather(self.model.pred, indices=tf.squeeze(pos_indices))
        return tf.reduce_mean(positives, axis=0)

    def generate_model(self, bn_decay=None, aggregation='two_means'):
        # i_preds = [None] * self.num_instances_per_bag
        self.model.generate_model(input_pl=self.input_pl, bn_decay=bn_decay, reuse=False)
        # i_preds[i] = tf.expand_dims(self.model.pred, 1)
        # instances = tf.concat(i_preds, 1)
        # print instances
        ##### Aggregation #####

        ### MAX ###
        if aggregation == 'max':
            self.pred = self.model.pred[tf.cast(tf.argmax(self.model.pred[:, 1]), tf.int32)]
            self.pred = tf.expand_dims(self.pred, axis=0)
            self.logits = self.pred
        ###########

        # self.features = tf.concat([tf.reduce_mean(instances, axis=1, keep_dims=True), tf.reduce_prod(instances, axis=1, keep_dims=True),
        #                            tf.reduce_max(instances, axis=1, keep_dims=True), tf.reduce_min(instances, axis=1, keep_dims=True),
        #                            tf.reduce_sum(instances, axis=1, keep_dims=True)], 1, name='mil_features')

        ### NOISY_AND ###
        if aggregation == 'noisy_and':
            with tf.variable_scope(type(self).__name__):
                self.prob = tf.expand_dims(self.model.pred, axis=0)
                self.logits = nn_layers.noisy_and_1d(self.prob, 2 if self.use_softmax else 1)
                # self.logits = nn_layers.fc(self.prob, 2, 2, 'predicted_y_mil', is_training=self.is_training, activation_fn=None)
                self.pred = tf.nn.softmax(self.logits, name='softmax_mil')
        #################

        ### TWO_MEANS ###
        if aggregation == 'two_means':
            # self.mean = tf.cond(tf.reduce_all(tf.equal(self.label_pl, [[0., 1.]])), # True should be when label is positive
            #                     fn1=lambda: self.get_positive_mean(self.model.pred[:, 1]), # Get mean of instances grouped into the positive cluster
            #                     fn2=lambda: tf.reduce_mean(self.model.pred, axis=0)) # Get mean across all instances
            self.mean = self.get_positive_mean(self.model.pred[:, 1])
            self.logits = tf.expand_dims(self.mean, axis=0)
            self.pred = tf.nn.softmax(self.logits, name='softmax_mil')
        #################

        ### PRE_CLUSTER ###
        if aggregation == 'feature_cluster':
            self.pos_cluster_idx = self.cluster()
            self.mean = self.reduce_mean(tf.gather(self.model.pred, self.pos_cluster_idx), axis=0)
            self.logits = tf.expand_dims(self.mean, axis=0)
            self.pred = tf.nn.softmax(self.logits, name='softmax_mil')
        ###################

        self.model.generate_model(bn_decay=bn_decay, reuse=True)
        return self.pred

    def generate_loss(self):
        if self.use_softmax:
            # loss_p = tf.nn.softmax_cross_entropy_with_logits(logits=self.prob, labels=self.label_pl,
            #                                                  name='sigmoid_xentropy_mil')
            loss_l = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label_pl,
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
            labels = np.array([[0., 1.]]) if self.use_softmax else [1.]
            yield data, labels

            data = neg[0]
            labels = np.array([[1., 0.]]) if self.use_softmax else [0.]
            yield data, labels

    def save(self, sess, model_path, global_step=None):
        self.model.save(sess, model_path, global_step=global_step)
        # super(SegmentedMIL, self).save(sess, model_path, global_step=global_step)

    def restore(self, sess, model_path):
        self.model.restore(sess, model_path)
        # super(SegmentedMIL, self).restore(sess, model_path)
