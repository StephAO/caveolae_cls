import os
import numpy as np
import scipy.io as sio
from caveolae_cls.data_handler import DataHandler


class CNNDataHandler(DataHandler):

    def __init__(self, input_data_type):
        super(CNNDataHandler, self).__init__()
        if input_data_type == "multiview" or input_data_type == "projections":
            self.data_key = 'Img3Ch'
            self.files = '/home/stephane/Dropbox/Data_Stephane/BlobsProjectionsImg_Exp4_MAT_3Channels'
        self.train_files = self.files[:int(0.9 * len(self.files))]
        self.eval_files = self.files[int(0.9 * len(self.files)):]

    def load_input_data(self, filename):
        """
        Load point cloud data and label given a file
        """
        f = sio.loadmat(filename)
        data = f[self.data_key][:]
        label = DataHandler.get_label_from_filename(filename)
        return data, label

    def get_batch(self, batch_shape, max_ratio_n):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param max_ratio_n: Maximum ratio of negative data points in a single batch.
                            A value of 1 would mean that a batch containing only
                            negative elements would be acceptable.
        :param batch_shape: Expected shape of a single batch
        :return: Generates batches
        """
        self.batch_size = batch_shape[0]
        self.data = np.zeros(batch_shape)
        self.labels = np.zeros([self.batch_size])

        if eval:
            files = self.eval_files
        else:
            files = self.train_files

        random_file_idxs = np.arange(len(files))
        np.random.shuffle(random_file_idxs)

        i = 0
        num_negatives = 0
        for idx in random_file_idxs:
            f = files[idx]
            d, l = self.load_input_data(f)
            if l == 0:
                if num_negatives >= int(max_ratio_n * self.batch_size):
                    continue
                num_negatives += 1
            self.data[i] = d
            self.labels[i] = l
            i += 1
            if i >= self.batch_size:
                # Yield batch
                yield self.data, self.labels
                i = 0
                num_negatives = 0


