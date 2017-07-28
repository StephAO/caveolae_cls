from abc import ABCMeta, abstractmethod
import numpy as np
import os


class DataHandler:
    __metaclass__ = ABCMeta

    proj_dim = 512
    feature_shape = [32, 32, 32]

    def __init__(self, p_file_dir, n_file_dir, p_file_dir_inst=None, n_file_dir_inst=None, use_softmax=False):
        self.batch_size = None
        self.data = None
        self.labels = None
        self.use_softmax = use_softmax

        if not isinstance(p_file_dir, list) and not isinstance(n_file_dir, list):
            self.p_train_files = DataHandler.get_data_files(os.path.join(p_file_dir, "training"))
            self.p_eval_files = DataHandler.get_data_files(os.path.join(p_file_dir, "validation"))
            self.n_train_files = DataHandler.get_data_files(os.path.join(n_file_dir, "training"))[:len(self.p_train_files)]
            self.n_eval_files = DataHandler.get_data_files(os.path.join(n_file_dir, "validation"))[:len(self.p_eval_files)]
        else:
            self.bag = {'pos': {'train': {}, 'val': {}, 'test': {}}, 'neg': {'train': {}, 'val': {}, 'test': {}}}
            self.inst = {'pos': {'train': {}, 'val': {}, 'test': {}}, 'neg': {'train': {}, 'val': {}, 'test': {}}}

    @staticmethod
    def get_fucking_files(directories, storage_dict, include_synthetics=True):
        for d in directories:
            pos_sub_dir = os.path.join(d, "Projs_" + d.split('_')[1] + "_MAT_PC3")
            neg_sub_dir = os.path.join(d, "Projs_" + d.split('_')[1] + "_MAT_PC3PTRF")
            for f in os.listdir(pos_sub_dir):

                if (os.path.isfile(os.path.join(pos_sub_dir, f)) and
                        (include_synthetics or f.split('_')[0] != 'synthetic')):
                    exp_num = d.split('_')[1]
                    cell_num = f.split('_')[1]

                    storage_dict['pos']['train']

                    token = exp_num + '_' + cell_num
                    if cell_num == '1':
                        storage_dict['pos']['test'][token] = os.path.isfile(os.path.join(pos_sub_dir, f))
                    elif cell_num == '2':
                        storage_dict['pos']['val'][token] = os.path.isfile(os.path.join(pos_sub_dir, f))
                    else:
                        storage_dict['pos']['train'][token] = os.path.isfile(os.path.join(pos_sub_dir, f))


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


