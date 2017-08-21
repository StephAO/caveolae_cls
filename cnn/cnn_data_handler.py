import numpy as np
import scipy.io as sio
import sys

from caveolae_cls.data_handler import DataHandler

class CNNDataHandler(DataHandler):

    def __init__(self, input_data_type, num_classes=2, use_softmax=False):
        self.input_data_type = input_data_type
        if input_data_type == "multiview" or input_data_type == "projection":
            self.data_key = 'Img3Ch'
        self.num_classes = num_classes
        super(CNNDataHandler, self).__init__('Projs', use_softmax=use_softmax)

    def load_input_data(self, filename):
        """
        Load point cloud data and label given a file
        """
        f = sio.loadmat(filename)
        data = f[self.data_key][:]
        # if np.count_nonzero(data) < 60:
        #     return None, None
        label = DataHandler.get_label_from_filename(filename)
        if self.use_softmax:
            l = np.zeros([self.num_classes])
            # num_points = np.count_nonzero(data)
            # if label == 1:
            #     l[0] = 1
            # elif num_points <= 5:
            #     l[1] = 1
            # elif num_points <= 10:
            #     l[2] = 1
            # elif num_points <= 15:
            #     l[3] = 1
            # elif num_points <= 20:
            #     l[4] = 1
            # elif num_points <= 25:
            #     l[5] = 1
            # elif num_points <= 30:
            #     l[6] = 1
            # elif num_points <= 50:
            #     l[7] = 1
            # elif num_points <= 100:
            #     l[8] = 1
            # else:
            #     l[9] = 1

            l[label] = 1
            label = l
        return data, label

    def get_batch(self, batch_shape, use='train', val_set=None, cell_type=None, verbose=True,):
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
        self.labels = np.zeros([self.batch_size, self.num_classes] if self.use_softmax else self.batch_size)

        files = self.get_data_files(use=use, val_set=val_set, cell_type=cell_type)
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
            if d is None:
                continue
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
