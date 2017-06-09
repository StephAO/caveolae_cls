import numpy as np
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler


class CNNDataHandler(DataHandler):

    def __init__(self, input_data_type):
        super(CNNDataHandler, self).__init__()
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'
            self.p_files = DataHandler.get_data_file(
                '/staff/2/sarocaou/data/projection_positive')
            self.n_files = DataHandler.get_data_file(
                '/staff/2/sarocaou/data/projection_negative')[:len(self.p_files)]
        self.p_train_files = self.p_files[:int(0.9 * len(self.p_files))]
        self.p_eval_files = self.p_files[int(0.9 * len(self.p_files)):]
        self.n_train_files = self.n_files[:int(0.9 * len(self.n_files))]
        self.n_eval_files = self.n_files[int(0.9 * len(self.n_files)):]

    def load_input_data(self, filename):
        """
        Load point cloud data and label given a file
        """
        f = sio.loadmat(filename)
        data = f[self.data_key][:]
        label = DataHandler.get_label_from_filename(filename)
        return data, label

    def get_batch(self, batch_shape, eval=False):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param max_ratio_n: Maximum ratio of negative data points in a single batch.
                            A value of 1 would mean that a batch containing only
                            negative elements would be acceptable.
        :param batch_shape: Expected shape of a single batch
        :return: Generates batches
        """
        print batch_shape
        self.batch_size = batch_shape[0]
        self.data = np.zeros(batch_shape)
        self.labels = np.zeros([self.batch_size])

        if eval:
            p_files = self.p_eval_files
            n_files = self.n_eval_files
        else:
            p_files = self.p_train_files
            n_files = self.n_train_files

        p_random_file_idxs = np.arange(len(p_files))
        np.random.shuffle(p_random_file_idxs)

        n_random_file_idxs = np.arange(len(n_files))
        np.random.shuffle(n_random_file_idxs)

        random_file_idxs = zip(p_random_file_idxs, n_random_file_idxs)

        i = 0
        # num_negatives = 0
        progress = 0
        for count, idxs in enumerate(random_file_idxs):
            if float(count) / len(random_file_idxs) >= progress + 0.05:
                progress += 0.05
                print str(int(progress * 100)) + "%",
                sys.stdout.flush()
                if progress == 0.95:
                    print ""
            p_idx, n_idx = idxs
            p_f = p_files[p_idx]
            d, l = self.load_input_data(p_f)
            # if l == 0:
            #     if num_negatives >= int(max_ratio_n * self.batch_size):
            #         continue
            #     num_negatives += 1
            self.data[i] = d
            self.labels[i] = l

            n_f = n_files[n_idx]
            d, l = self.load_input_data(n_f)
            self.data[i + 1] = d
            self.labels[i + 1] = l
            i += 2
            if i >= self.batch_size:
                # Yield batch
                yield self.data, self.labels
                i = 0
                num_negatives = 0


