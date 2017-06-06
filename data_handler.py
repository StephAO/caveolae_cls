from abc import ABCMeta, abstractmethod
import os


class DataHandler:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.batch_size = None
        self.data = None
        self.labels = None

    @staticmethod
    def get_data_file(directory):
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
    def get_batch(self):
        pass


