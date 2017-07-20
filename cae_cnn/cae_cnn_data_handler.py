import numpy as np
import os
import scipy.io as sio
from scipy import ndimage
import sys
import tensorflow as tf

from caveolae_cls.data_handler import DataHandler
import caveolae_cls.cae.cae_model as cae


class CAE_CNN_DataHandler(DataHandler):

    def __init__(self, input_data_type, input_shape, use_softmax=False):
        self.input_data_type = input_data_type
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'
            # p_file_dir = '/staff/2/sarocaou/data/projection_positive'
            # n_file_dir = '/staff/2/sarocaou/data/projection_negative'
            p_file_dir_val = '/home/stephane/sfu_data/projection_positive'
            n_file_dir_val = '/home/stephane/sfu_data/projection_negative'
            p_file_dir = '/home/stephane/sfu_data/mil_data/positive'
            n_file_dir = '/home/stephane/sfu_data/mil_data/negative'

        super(CAE_CNN_DataHandler, self).__init__(p_file_dir, n_file_dir, use_softmax)

        self.p_eval_files = DataHandler.get_data_files(os.path.join(p_file_dir_val, "validation"))
        self.n_eval_files = DataHandler.get_data_files(os.path.join(n_file_dir_val, "validation"))[:len(self.p_eval_files)]

        self.cae = cae.CAE(input_data_type)
        self.cae_pl = None
        self.input_shape = input_shape
        self.features = None
        self.replicator = None
        self.sess = None
        self.model_save_path = None

    def generate_cae_placeholders(self):
        self.cae_pl = tf.placeholder(tf.float32, shape=self.input_shape)

    def generate_cae(self):
        if self.features is None or self.replicator is None:
            self.cae.generate_model(self.cae_pl)
            # try:
            self.cae.restore(self.sess, os.path.join(self.model_save_path))

            # except:
            #     print "Unable to load CAE model"
            #     exit()

            self.features = self.cae.features
            self.replicator = self.cae.pred

    def get_batch(self, batch_shape, eval=False, type='mixed'):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param batch_shape: Expected shape of a single batch
        :return: Generates batches
        """
        batch_size = batch_shape[0]
        sub_batch_size = self.input_shape[0]
        assert batch_size % sub_batch_size == 0

        data = np.zeros([sub_batch_size] + self.input_shape[1:])
        features = np.zeros(batch_shape)
        labels = np.zeros([batch_size, 2] if self.use_softmax else batch_size)

        files = []

        if eval:
            if type == 'mixed' or type == 'positive':
                files.extend(self.p_eval_files)
            if type == 'mixed' or type == 'negative':
                files.extend(self.n_eval_files)
        else:
            if type == 'mixed' or type == 'positive':
                files.extend(self.p_train_files)
            if type == 'mixed' or type == 'negative':
                files.extend(self.n_train_files)

        random_file_idxs = np.arange(len(files))
        np.random.shuffle(random_file_idxs)

        i = 0
        # num_negatives = 0
        progress = 0
        for count, idx in enumerate(random_file_idxs):
            if float(count) / len(random_file_idxs) >= progress + 0.05:
                progress += 0.05
                print str(int(round(progress * 100))) + "%",
                sys.stdout.flush()
                if abs(progress - 0.95) <= 0.01:
                    print ""
            f = files[idx]
            d, l = self.cae.data_handler.load_input_data(f, softmax=self.use_softmax)
            # if l == 0:
            #     if num_negatives >= int(max_ratio_n * self.batch_size):
            #         continue
            #     num_negatives += 1
            data[i % sub_batch_size] = d
            labels[i] = l

            i += 1
            if i % sub_batch_size == 0:
                # Yield batch
                features[i - sub_batch_size: i] = self.sess.run(self.cae.features, feed_dict={self.cae_pl: data})

            if i >= batch_size:
                yield features, labels
                i = 0
                # num_negatives = 0
