from abc import ABCMeta, abstractmethod
from pkg_resources import resource_filename
import yaml


class Model:
    __metaclass__ = ABCMeta

    def __init__(self, hp_fn="default.yaml"):
        self.hp = {}
        hyper_param_fullpath = resource_filename('caveolae_cls', hp_fn)
        self.load_hyperparams(hyper_param_fullpath)
        self.data_handler = None

    def load_hyperparams(self, filename):
        with open(filename, 'r') as f:
            try:
                self.hp = yaml.load(f)
            except yaml.YAMLError as e:
                print e
                exit()

    @abstractmethod
    def get_batch(self, eval=False, type='mixed'):
        pass

    @abstractmethod
    def get_input_placeholders(self):
        pass

    @abstractmethod
    def get_model(self, data_pl, is_training, bn_decay=None, reuse=None):
        pass

    @abstractmethod
    def get_loss(self, pred, label):
        pass