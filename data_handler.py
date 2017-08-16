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

    def __init__(self, input_type, use_softmax=True):
        self.batch_size = None
        self.data = None
        self.labels = None
        self.use_softmax = use_softmax

        self.train_group = 0
        self.num_groups = 10
        self.input_type = input_type
        self.inst = {}
        directories = ['/home/stephane/sfu_data/DL_Exp1', '/home/stephane/sfu_data/DL_Exp2',
                       '/home/stephane/sfu_data/DL_Exp3', '/home/stephane/sfu_data/DL_Exp4']
        self.sort_data_files(directories)
        self.shuffle_inst_files()


    def fill_data_dict(self, sub_dir, exp_num, bag_label, include_synthetics=False, exclude_PC3PTRF_neg=True):
        """
        Fills data dictionary for use later
        :param sub_dir: Directory containing only data files (lowest level directory)
        :param exp_num: Experiment number (1-4_
        :param include_synthetics: Boolean to include synthetic data if it exists
        :return: None
        """
        data_files = os.listdir(sub_dir)
        np.random.shuffle(data_files)
        for f in data_files:
            if (os.path.isfile(os.path.join(sub_dir, f)) and
                    (include_synthetics or f.split('_')[0] != 'synthetic')):
                filename_label = DataHandler.get_label_from_filename(f)
                if exp_num != 4:
                    recursive_dict_filler(self.inst, ['test', filename_label], os.path.join(sub_dir, f))
                elif exclude_PC3PTRF_neg and filename_label == 0 and sub_dir.split('_')[-1] == "PC3PTRF":
                    continue
                elif not exclude_PC3PTRF_neg and sub_dir.split('_')[-1] == "PC3":
                    continue
                else:
                    recursive_dict_filler(self.inst, ['train', self.train_group, filename_label], os.path.join(sub_dir, f))
                    self.train_group = (self.train_group + 1) % self.num_groups

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
        for d in directories:
            exp_num = int(d.split('/')[-1].split('_')[1][-1])
            pos_sub_dir = os.path.join(d, self.input_type + "_Exp" + str(exp_num) + "_MAT_PC3PTRF")
            neg_sub_dir = os.path.join(d, self.input_type + "_Exp" + str(exp_num) + "_MAT_PC3")
            self.fill_data_dict(pos_sub_dir, exp_num, 1)
            self.fill_data_dict(neg_sub_dir, exp_num, 0)


    def get_data_files(self, use, val_set=None, cell_type=None):
        """
        Return requested data files.
        WARNING: returned file list is in a mutable (does not copy for performance reasons)
        :param use: What the data files will be used for ('train', 'val', or 'test')
        :param val_set: Training data group being used for validation in current round of cross validation
        :return: requested data files
        """
        if use == 'test':
            if cell_type is None:
                files = self.inst[use][1] + self.inst[use][0][:len(self.inst[use][1])]
            elif cell_type == 'PC3':
                files = self.inst[use][0][:10000]
            elif cell_type == 'PC3PTRF':
                files = self.inst[use][1][:10000]
            else:
                raise TypeError("%s is an unknown cell type" % (cell_type))
        # elif use == 'val':
        #     p_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_positive', "validation"))
        #     n_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_negative', "validation"))[:len(p_files)]
        #     files = p_files + n_files
        # elif use == 'train':
        #     p_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_positive', "training"))
        #     n_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_negative', "training"))[:len(p_files)]
        #     files = p_files + n_files
        else:
            files = []
            for i in xrange(self.num_groups):
                if i == val_set and use == 'val':
                    files += self.inst['train'][i][1] + self.inst['train'][i][0][:len(self.inst['train'][i][1])]
                elif i != val_set and use == 'train':
                    files += self.inst['train'][i][1] + self.inst['train'][i][0][:len(self.inst['train'][i][1])]

        print "Using %d files" % len(files)
        return files

    def shuffle_inst_files(self):
        for use in self.inst:
            pos_count = 0
            neg_count = 0
            for i in self.inst[use]:
                if use == 'train':
                    for j in self.inst[use][i]:
                        pos_count += 0 if j == 0 else len(self.inst[use][i][j])
                        neg_count += 0 if j == 1 else len(self.inst[use][i][j])
                        np.random.shuffle(self.inst[use][i][j])
                else:
                    np.random.shuffle(self.inst[use][i])
            print pos_count
            print neg_count

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

    # @abstractmethod
    # def get_batch(self, batch_shape, use='train', label=None):
    #     pass


