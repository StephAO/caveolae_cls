import numpy as np
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler


class CNNDataHandler(DataHandler):

    def __init__(self, input_data_type, use_softmax=False, use_mil=False):
        self.input_data_type = input_data_type
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'

        super(CNNDataHandler, self).__init__(use_softmax=use_softmax, use_mil=use_mil)

    def load_input_data(self, filename):
        """
        Load point cloud data and label given a file
        """
        f = sio.loadmat(filename)
        data = f[self.data_key][:]
        label = DataHandler.get_label_from_filename(filename)
        if self.use_softmax:
            l = np.zeros([2])
            l[label] = 1
            label = l
        return data, label

    def get_batch(self, batch_shape, use='train', label=None, exp_cell_token=None, verbose=True):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param batch_shape: Expected shape of a single batch
        :param use: What batch will be used for (options: 'train', 'val', 'test')
        :param label
        :return: Generates batches
        """
        self.batch_size = batch_shape[0]
        self.data = np.zeros(batch_shape)
        self.labels = np.zeros([self.batch_size, 2] if self.use_softmax else self.batch_size)

        files = self.get_data_files(use=use, label=label, exp_cell_token=exp_cell_token)
        
        print "Using %d files" % len(files)

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
            self.data[i] = d
            self.labels[i] = l

            i += 1
            if i >= self.batch_size:
                # Yield batch
                yield self.data, self.labels
                i = 0
                # num_negatives = 0
