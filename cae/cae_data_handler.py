import numpy as np
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler


class CAEDataHandler(DataHandler):

    def __init__(self, input_data_type, use_softmax=False):
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'
            p_file_dir = '/staff/2/sarocaou/data/projection_positive'
            n_file_dir = '/staff/2/sarocaou/data/projection_negative'
        super(CAEDataHandler, self).__init__(p_file_dir, n_file_dir, use_softmax)

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

    def get_batch(self, batch_shape, eval=False, type='mixed'):
        """
        Generator that will return batches
        :param files: List of data file names. Each file should contain a 1 element.
        :param batch_shape: Expected shape of a single batch
        :return: Generates batches
        """
        self.batch_size = batch_shape[0]
        self.input = np.zeros(batch_shape)

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


