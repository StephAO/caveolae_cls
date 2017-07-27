import tensorflow as tf
import numpy as np

class K_Means:

    def __init__(self, num_samples, num_clusters):
        self.num_clusters = num_clusters
        self.num_samples = num_samples
        self.samples = None

    def choose_centroids(self, first_centroid_idx):
        # Uses k-means++ algorithm
        if first_centroid_idx is None:
            first_centroid_idx = tf.random_uniform((), minval=0, maxval=self.num_samples + 1, dtype=tf.int32)
        centroids = []
        centroids.append(tf.expand_dims(self.samples[0][first_centroid_idx], 0))

        for i in xrange(1, self.num_clusters):
            distances_to_others = tf.reduce_sum(tf.square(tf.subtract(self.samples, centroids)), axis=(2, 3, 4))
            min_distance = tf.reduce_min(distances_to_others, axis=0)
            prob_dist = tf.divide(min_distance, tf.reduce_sum(min_distance))
            cum_prob = tf.cumsum(prob_dist)
            rand = tf.random_uniform((1, ))
            new_centroid_index = tf.where(cum_prob >= rand)[0][0]
            centroids.append(tf.expand_dims(self.samples[0][tf.to_int32(new_centroid_index)], 0))

        centroids = tf.concat([tf.expand_dims(centroid, 0) for centroid in centroids], 0)
        # print tf_centroids.get_shape().as_list()
        return centroids

    def assign_to_nearest(self, centroids):
        # Finds the nearest centroid for each sample

        # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
        distances = tf.reduce_sum(tf.square(tf.subtract(self.samples, centroids)), axis=(2, 3, 4))
        mins = tf.argmin(distances, 0)
        # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
        nearest_indices = mins
        return nearest_indices

    def update_centroids(self, nearest_indices):
        # Updates the centroid to be the mean of all samples associated with it.
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(tf.squeeze(self.samples), nearest_indices, self.num_clusters)
        new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0, keep_dims=True), 0) for partition in partitions], 0)
        return new_centroids

    def cluster(self, samples, highest_pos_idx, num_iterations=10):
        self.samples = tf.expand_dims(samples, 0)
        centroids = self.choose_centroids(highest_pos_idx)
        # centroids = tf.concat([tf.expand_dims(self.samples[highest_pos_idx], axis=0),
        #                        tf.expand_dims(self.samples[highest_neg_idx], axis=0)], 0)
        nearest_indices = self.assign_to_nearest(centroids)
        for i in xrange(num_iterations):
            centroids = self.update_centroids(nearest_indices)
            nearest_indices = tf.to_float(self.assign_to_nearest(centroids))
        # since index will either be 0 or 1, if more than half of indices are 1, then the rounded mean will be 1
        # smaller_cluster = tf.rint(tf.reduce_mean(nearest_indices))
        in_pos_cluster = tf.equal(nearest_indices, 0.)
        input_idx_in_cluster = tf.where(in_pos_cluster)
        return input_idx_in_cluster #, nearest_indices