from abc import ABCMeta, abstractmethod
import numpy as np
import os

def recursive_dict_filler(dict_, keys, item):
    """
    Fills dictionary if dictionary structure exists, else recursively creates missing dictionaries and then fills it
    :param dict: base dictionary
    :param keys: list of keys
    :return: None, dict is filled in place
    """
    if len(keys) == 1:
        try:
            dict_[keys[0]].append(item)
        except KeyError:
            dict_[keys[0]] = [item]
    else:
        try:
            deeper_dict = dict_[keys[0]]
        except KeyError:
            dict_[keys[0]] = {}
            deeper_dict = dict_[keys[0]]
        recursive_dict_filler(deeper_dict, keys[1:], item)


class DataHandler:
    __metaclass__ = ABCMeta

    proj_dim = 512
    feature_shape = [32, 32, 32]

    def __init__(self, use_softmax=False, use_organized_data=False):
        self.batch_size = None
        self.data = None
        self.labels = None
        self.use_softmax = use_softmax
        self.use_organized_data = use_organized_data

        if use_organized_data:
            self.bag = {}
            self.inst = {}
            directories = ['/home/stephane/sfu_data/DL_Exp1', '/home/stephane/sfu_data/DL_Exp2',
                           '/home/stephane/sfu_data/DL_Exp3', '/home/stephane/sfu_data/DL_Exp4']
            self.sort_data_files(directories)
            self.shuffle_inst_files()
        else:
            # p_file_dir = '/staff/2/sarocaou/data/projection_positive'
            # n_file_dir = '/staff/2/sarocaou/data/projection_negative'
            p_file_dir = '/home/stephane/sfu_data/projection_positive'
            n_file_dir = '/home/stephane/sfu_data/projection_negative'

            p_train_files = DataHandler.get_data_filepaths(os.path.join(p_file_dir, "training"))
            n_train_files = DataHandler.get_data_filepaths(os.path.join(n_file_dir, "training"))[:len(p_train_files)]
            p_val_files = DataHandler.get_data_filepaths(os.path.join(p_file_dir, "validation"))
            n_val_files = DataHandler.get_data_filepaths(os.path.join(n_file_dir, "validation"))[:len(p_val_files)]
            p_test_files = DataHandler.get_data_filepaths(os.path.join(p_file_dir, "testing"))
            n_test_files = DataHandler.get_data_filepaths(os.path.join(n_file_dir, "testing"))[:len(p_test_files)]

            self.train_files = p_train_files + n_train_files
            self.val_files = p_val_files + n_val_files
            self.test_files = p_test_files + n_test_files


    def fill_data_dict(self, sub_dir, exp_num, label, val_cells, include_synthetics=True, exclude_PC3PTRF_neg=True):
        """
        Fills data dictionary for use later
        :param sub_dir: Directory containing only data files (lowest level directory)
        :param exp_num: Experiment number (1-4_
        :param label: 'pos' or 'neg' based on whether it's coming from PC3 (neg) or PC3PTRF (pos)
        :param val_cells: randomly selected cells used for validation (range 3-9 inclusive)
        :param include_synthetics: Boolean to include synthetic data if it exists
        :return: None
        """
        for f in os.listdir(sub_dir):
            if (os.path.isfile(os.path.join(sub_dir, f)) and
                    (include_synthetics or f.split('_')[0] != 'synthetic')):
                cell_num = int(f.split('_')[1])
                if cell_num == 1 or (exp_num == 4 and cell_num == 2):
                    use = 'test'
                elif cell_num == val_cells[exp_num - 1] or (exp_num == 4 and cell_num == val_cells[exp_num - 1] + 1):
                    use = 'val'
                else:
                    use = 'train'

                recursive_dict_filler(self.bag, [use, label, 100 * exp_num + cell_num], os.path.join(sub_dir, f))
                filename_label = DataHandler.get_label_from_filename(f)
                if exclude_PC3PTRF_neg and filename_label == 0 and sub_dir.split('_')[-1] == "PC3PTRF":
                    continue
                elif exp_num == 4:
                    recursive_dict_filler(self.inst, [use, filename_label], os.path.join(sub_dir, f))


    def sort_data_files(self, directories):
        """
        Assuming directorry strucuture is as follows:
            directories[i] = "DL_Expi
                Projs_Expi_MAT_PC3
                    Cell_j_Blob_k_label_l.mat
                    ...
                Projs_Expi_MAT_PC3PTRF
                    Cell_j_Blob_k_label_l.mat
                    ...

        :param directories: Directories of experiments containing data
        :return:
        """
        val_cells = np.random.randint(2, 10, size=4)
        for d in directories:
            exp_num = int(d.split('/')[-1].split('_')[1][-1])
            pos_sub_dir = os.path.join(d, "Projs_Exp" + str(exp_num) + "_MAT_PCA_PC3PTRF")
            neg_sub_dir = os.path.join(d, "Projs_Exp" + str(exp_num) + "_MAT_PCA_PC3")
            # pos_sub_dir = os.path.join(d, "Projs_Exp" + str(exp_num) + "_MAT_PC3PTRF")
            # neg_sub_dir = os.path.join(d, "Projs_Exp" + str(exp_num) + "_MAT_PC3")
            self.fill_data_dict(pos_sub_dir, exp_num, 'pos', val_cells)
            self.fill_data_dict(neg_sub_dir, exp_num, 'neg', val_cells)


    def get_data_files(self, use, label=None, exp_cell_token=None, get_all=False):
        """
        Return requested data files.
        WARNING: returned file list is in a mutable (not copying for performance reasons)
        :param use: What the data files will be used for ('train', 'val', or 'test')
        :param label: Only used for mil. 'pos' or 'neg' will return the positive/negative cell
                      associated with exp_cell_token. If None, return instances.
        :param exp_cell_token: Only used if use_organized_data if False and label is not None. Identifies specific cell to return.
        :param get_all: Returns all files for that use
        :return: requested data files
        """
        if self.use_organized_data:
            if get_all:
                files = []
                for l in self.bag[use]:
                    for token in self.bag[use][l]:
                        files += self.bag[use][l][token]
            elif label is None:
                files = self.inst[use][1] + self.inst[use][0][:len(self.inst[use][1])]
            else:
                files = self.bag[use][label][exp_cell_token]
        else:
            if use == 'train':
                files = self.train_files
            elif use == 'val':
                files = self.val_files
            elif use == 'test':
                files = self.p_test_files

        return files

    def shuffle_inst_files(self):
        for use in self.inst:
            for l in self.inst[use]:
                np.random.shuffle(self.inst[use][l])

    @staticmethod
    def get_data_filepaths(directory, include_synthetics=True):
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
    def get_batch(self, batch_shape, use='train', label=None):
        pass


