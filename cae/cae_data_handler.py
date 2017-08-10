import numpy as np
import os
import scipy.io as sio
from scipy import ndimage
import sys

from caveolae_cls.data_handler import DataHandler


class CAEDataHandler(DataHandler):

    def __init__(self, input_data_type, use_organized_data=True):
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'
        super(CAEDataHandler, self).__init__(use_organized_data=use_organized_data)

    def load_input_data(self, filename, softmax=True):
        """
        Load point cloud data and label given a file
        """
        f = sio.loadmat(filename)
        data = f[self.data_key][:]
        for i in xrange(data.shape[-1]):
            data[:, :, i] = ndimage.distance_transform_edt(1 - data[:, :, i])
        label = DataHandler.get_label_from_filename(filename)
        if softmax:
            l = np.zeros([2])
            l[label] = 1
            label = l
        return data, label

    def get_batch(self, batch_shape, use='train', label=None, exp_cell_token=None, verbose=True):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param batch_shape: Expected shape of a single batch
        :return: Generates batches
        """
        self.batch_size = batch_shape[0]
        self.input = np.zeros(batch_shape)

        files = self.get_data_files(use=use, label=label, exp_cell_token=exp_cell_token, get_all=True)

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
            d, l = self.load_input_data(f)
            # if l == 0:
            #     if num_negatives >= int(max_ratio_n * self.batch_size):
            #         continue
            #     num_negatives += 1
            self.input[i] = d

            i += 1
            if i >= self.batch_size:
                # Yield batch
                yield self.input, None
                i = 0
                # num_negatives = 0






















