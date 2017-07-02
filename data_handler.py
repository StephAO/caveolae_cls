from abc import ABCMeta, abstractmethod
import numpy as np
import os


class DataHandler:
    __metaclass__ = ABCMeta

    proj_dim = 512
    feature_shape = [32, 32, 16]

    def __init__(self, p_file_dir, n_file_dir, use_softmax=False):
        self.batch_size = None
        self.data = None
        self.labels = None
        self.use_softmax = use_softmax

        self.p_train_files = DataHandler.get_data_files(os.path.join(p_file_dir, "training"))
        self.p_eval_files = DataHandler.get_data_files(os.path.join(p_file_dir, "validation"))
        self.n_train_files = DataHandler.get_data_files(os.path.join(n_file_dir, "training"))[:len(self.p_train_files)]
        self.n_eval_files = DataHandler.get_data_files(os.path.join(n_file_dir, "validation"))[:len(self.p_eval_files)]

    @staticmethod
    def get_data_files(directory, include_synthetics=True):
        """ Return all files in a given directory """
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if (os.path.isfile(os.path.join(directory, f)) and
                    (include_synthetics or f.split('_')[0] != 'synthetic'))]

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


