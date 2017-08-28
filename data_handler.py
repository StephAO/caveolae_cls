from abc import ABCMeta, abstractmethod
import numpy as np
import os

def recursive_dict_filler(dict_, keys, item=None):
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
            if item is None:
                dict_[keys[0]] = []
            else:
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

    def __init__(self, input_type, use_softmax=True, xval=False, cell_division=False):
        self.batch_size = None
        self.data = None
        self.labels = None
        self.use_softmax = use_softmax
        self.cell_division = cell_division
        self.xval = xval

        self.train_group = 0
        if self.cell_division:
            self.groups = [1, 2, 3, 5, 6, 8, 9, 10]
        else:
            self.groups = range(10)
        self.num_groups = len(self.groups)
        self.input_type = input_type
        self.inst = {}
        directories = ['/home/stephane/sfu_data/DL_Exp4']
        self.sort_data_files(directories)
        self.shuffle_inst_files()


    def fill_data_dict(self, sub_dir, exp_num, bag_label, include_synthetics=False, exclude_PC3PTRF_neg=False):
        """
        Fills data dictionary for use later
        :param sub_dir: Directory containing only data files (lowest level directory)
        :param exp_num: Experiment number (1-4_
        :param include_synthetics: Boolean to include synthetic data if it exists
        :return: None
        """
        data_files = os.listdir(sub_dir)
        for f in data_files:
            cell_num = int(f.split('_')[1])
            if cell_num == 7:
                cell_num = 6
            if (os.path.isfile(os.path.join(sub_dir, f)) and
                    (include_synthetics or f.split('_')[0] != 'synthetic')):
                filename_label = DataHandler.get_label_from_filename(f)
                if self.xval:
                    group_label = cell_num if self.cell_division else self.train_group
                    recursive_dict_filler(self.inst, ['train', group_label, filename_label], os.path.join(sub_dir, f))
                    self.train_group = (self.train_group + 1) % self.num_groups
                # elif exclude_PC3PTRF_neg and filename_label == 0 and sub_dir.split('_')[-1] == "PC3PTRF":
                #     continue
                # elif not exclude_PC3PTRF_neg and sub_dir.split('_')[-1] == "PC3":
                #     continue
                else:
                    recursive_dict_filler(self.inst, ['all', cell_num, filename_label], os.path.join(sub_dir, f))

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

            if not self.xval:
                for i in xrange(2):
                    recursive_dict_filler(self.inst, ['test', i])
                    recursive_dict_filler(self.inst, ['val', i])
                    recursive_dict_filler(self.inst, ['train', i])
                    for cell in [1, 2, 3, 4, 5, 6, 8, 9, 10]:
                        if self.cell_division:
                            if cell == 1:
                                self.inst['test'][i] += self.inst['all'][cell][i][:]
                            elif cell == 2:
                                self.inst['val'][i] += self.inst['all'][cell][i][:]
                            else:
                                self.inst['train'][i] += self.inst['all'][cell][i][:]
                        else:
                            test_idx = (12 if cell == 1 else 11)
                            # print test_idx
                            val_idx = test_idx + (6 if cell <= 5 else 5)
                            self.inst['test'][i] += self.inst['all'][cell][i][:test_idx]
                            self.inst['val'][i] += self.inst['all'][cell][i][test_idx:val_idx]
                            self.inst['train'][i] += self.inst['all'][cell][i][val_idx:]

                print "Size of test set pos %d neg %d, size of val set pos %d, neg %d, size of train set pos %d, neg %d" \
                       % (len(self.inst['test'][1]), len(self.inst['test'][0]), len(self.inst['val'][1]), len(self.inst['val'][0]), len(self.inst['train'][1]), len(self.inst['train'][0]))
                del self.inst['all']


    def get_data_files(self, use, val_set=None, cell_type=None):
        """
        Return requested data files.
        WARNING: returned file list is in a mutable (does not copy for performance reasons)
        :param use: What the data files will be used for ('train', 'val', or 'test')
        :param val_set: Training data group being used for validation in current round of cross validation
        :return: requested data files
        """
        files = []
        if self.xval:
            for i in self.groups:
                if i == val_set and use == 'val':
                    files += self.inst['train'][i][1] + self.inst['train'][i][0][:len(self.inst['train'][i][1])]
                elif i != val_set and use == 'train':
                    files += self.inst['train'][i][1] + self.inst['train'][i][0][:len(self.inst['train'][i][1])]
        # elif use == 'val':
        #     p_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_positive', "validation"))
        #     n_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_negative', "validation"))[:len(p_files)]
        #     files = p_files + n_files
        # elif use == 'train':
        #     p_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_positive', "training"))
        #     n_files = DataHandler.get_data_filepaths(os.path.join('/home/stephane/sfu_data/projection_negative', "training"))[:len(p_files)]
        #     files = p_files + n_files
        else:
            files += self.inst[use][1] + self.inst[use][0][:len(self.inst[use][1])]

        self.shuffle_inst_files()
        print "Using %d files" % len(files)
        return files

    def shuffle_inst_files(self):
        for use in self.inst:
            for i in self.inst[use]:
                if self.xval:
                    for j in self.inst[use][i]:
                        np.random.shuffle(self.inst[use][i][j])
                else:
                    np.random.shuffle(self.inst[use][i])

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


