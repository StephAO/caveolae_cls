from abc import ABCMeta, abstractmethod
import os
from pkg_resources import resource_filename
import tensorflow as tf
import yaml



class Model:
    __metaclass__ = ABCMeta

    def __init__(self, hp_fn="default.yaml"):
        self.hp = {}
        hyper_param_fullpath = resource_filename('caveolae_cls', hp_fn)
        self.load_hyperparams(hyper_param_fullpath)
        self.data_handler = None
        self.saver = None
        self.loss = None
        self.val_loss = None

    def load_hyperparams(self, filename):
        with open(filename, 'r') as f:
            try:
                self.hp = yaml.load(f)
            except yaml.YAMLError as e:
                print e
                exit()

    def generate(self, bn_decay=None):
        self.generate_input_placeholders()
        self.generate_model(bn_decay=bn_decay)
        self.generate_loss()

    def save(self, sess, model_path, global_step=None):
        if self.saver is None:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                   scope=type(self).__name__))
        model_name = type(self).__name__
        save_path = os.path.join(model_path, model_name)
        save_path = self.saver.save(sess, save_path, global_step=global_step)
        print "Model saved to file: %s" % save_path

    def restore(self, sess, model_path):
        if self.saver is None:
            self.saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=type(self).__name__))
        most_recent_ckpt = tf.train.latest_checkpoint(model_path)
        self.saver.restore(sess, most_recent_ckpt)
        print "Model restored from file: %s" % most_recent_ckpt

    @abstractmethod
    def get_batch(self, eval=False, type='mixed'):
        pass

    @abstractmethod
    def generate_input_placeholders(self):
        pass

    @abstractmethod
    def generate_model(self, data_pl, is_training, bn_decay=None, reuse=None):
        pass

    @abstractmethod
    def generate_loss(self, pred, label):
        pass