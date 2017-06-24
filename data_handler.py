from abc import ABCMeta, abstractmethod
import numpy as np
import os


class DataHandler:
    __metaclass__ = ABCMeta

    def __init__(self, p_file_dir, n_file_dir):
        self.batch_size = None
        self.data = None
        self.labels = None

        self.p_files = DataHandler.get_data_files(p_file_dir)
        self.n_files = DataHandler.get_data_files(n_file_dir)

        np.random.shuffle(self.p_files)
        np.random.shuffle(self.n_files)
        self.p_train_files = self.p_files[:int(0.9 * len(self.p_files))]
        self.p_eval_files = self.p_files[int(0.9 * len(self.p_files)):]
        self.n_train_files = self.n_files[:int(0.9 * len(self.n_files))]
        self.n_eval_files = self.n_files[int(0.9 * len(self.n_files)):]

    @staticmethod
    def get_data_files(directory):
        """ Return all files in a given directory """
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))]

    @staticmethod
    def get_label_from_filename(filename):
        """
        Parse label from filename.
        Filename is expected to be in format: Cell_i_Blob_j_Label_l.MAT
        where l is either 'N' for negative or 'P' for positive
        """
        label = filename.split("/")[-1].split("_")[-1].strip(".mat").strip(" ")
        return 0 if label == 'N' else 1

    @abstractmethod
    def get_batch(self, batch_shape, eval=False):
        pass


