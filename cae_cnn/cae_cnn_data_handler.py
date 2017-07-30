import numpy as np
import os
import scipy.io as sio
from scipy import ndimage
import sys
import tensorflow as tf

from caveolae_cls.data_handler import DataHandler
import caveolae_cls.cae.cae_model as cae


class CAE_CNN_DataHandler(DataHandler):
    # Instance = inst
    # TODO REMEMBER about filtering blobs with <60 points
    def __init__(self, input_data_type, input_shape, use_softmax=True, use_mil=False):
        self.input_data_type = input_data_type
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'

        super(CAE_CNN_DataHandler, self).__init__(use_softmax=use_softmax, use_mil=use_mil)

        # # TODO CHECK YOUR FUCKING DIFF and CHANGE "training" back to "validation"
        # self.p_eval_files_instance = DataHandler.get_data_filepaths(os.path.join(p_file_dir_val, "validation"))
        # self.n_eval_files_instance = DataHandler.get_data_filepaths(os.path.join(n_file_dir_val, "validation"))[:len(self.p_eval_files_instance)]

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

    def get_batch(self, batch_shape, use='train', label=None, exp_cell_token=None, verbose=True):
        """
        Generator that will return batches
        :param batch_shape: shape of batch to return
        :param use: What the batch will be used for ('train', 'val', or 'test')
        :param label: Only used for mil (otherwise None). For mil, either 'pos' or 'neg'
        :param exp_cell_token: Unique identifier of cell. 100 * experiment number + cell number
        :param verbose: Print stuff or not
        :return: yields batches
        """
        batch_size = batch_shape[0]
        sub_batch_size = self.input_shape[0]
        assert batch_size % sub_batch_size == 0

        data = np.zeros([sub_batch_size] + self.input_shape[1:])
        features = np.zeros(batch_shape)
        labels = np.zeros([batch_size, 2] if self.use_softmax else batch_size)

        files = self.get_data_files(use=use, label=label, exp_cell_token=exp_cell_token)

        if len(files) < batch_size:
            print "Not enough files to create a batch"
            return

        random_file_idxs = np.arange(len(files))
        np.random.shuffle(random_file_idxs)

        i = 0
        # num_negatives = 0
        progress = 0
        for count, idx in enumerate(random_file_idxs):
            if verbose and float(count) / len(random_file_idxs) >= progress + 0.05:
                progress += 0.05
                print str(int(round(progress * 100))) + "%",
                sys.stdout.flush()
                if abs(progress - 0.95) <= 0.01:
                    print ""
            f = files[idx]
            d, l = self.cae.data_handler.load_input_data(f, softmax=self.use_softmax)
            data[i % sub_batch_size] = d
            labels[i] = l

            i += 1
            if i % sub_batch_size == 0:
                features[i - sub_batch_size: i] = self.sess.run(self.cae.features, feed_dict={self.cae_pl: data})

            if i >= batch_size:
                yield features, labels
                i = 0

                if len(random_file_idxs) - count < batch_size:
                    break
