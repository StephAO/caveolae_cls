import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time
from pkg_resources import resource_filename

import caveolae_cls.pointnet.pointnet_model as pn
import caveolae_cls.cnn.cnn_model as cnn
import caveolae_cls.cae.cae_model as cae
import caveolae_cls.cae_cnn.cae_cnn_model as cae_cnn
import caveolae_cls.segmented_mil.segmented_mil_model as segmented_mil
import caveolae_cls.subregion_mil.subregion_mil_model as subregion_mil


class Train:

    def __init__(self, FLAGS):
        self.flags = FLAGS
        # Select model
        if FLAGS.model == "pointnet":
            self.model = pn.PointNet(use_mil=(FLAGS.mil is not None))
            FLAGS.input_type = "pointcloud"
            self.classification = True
        elif FLAGS.model == "cnn":
            self.model = cnn.CNN(FLAGS.input_type, use_mil=(FLAGS.mil is not None))
            self.classification = True
        elif FLAGS.model == "cae":
            self.model = cae.CAE(FLAGS.input_type, use_mil=False)
            self.classification = False
        elif FLAGS.model == "cae_cnn":
            self.model = cae_cnn.CAE_CNN(FLAGS.input_type, use_mil=(FLAGS.mil is not None))
            self.classification = True
        else:
            raise NotImplementedError("%s is not an implemented model" % FLAGS.model)

        # Select mil
        if FLAGS.mil == "seg":
            self.model = segmented_mil.SegmentedMIL(self.model)
            self.mil = "seg"
        elif FLAGS.mil == "sub":
            self.model = subregion_mil.SubregionMIL()
            self.mil = "sub"
        else:
            self.mil = None

        # Other params
        self.gpu_index = FLAGS.gpu
        self.optimizer = FLAGS.optimizer

        self.batch_size = self.model.hp["BATCH_SIZE"]
        self.max_epoch = self.model.hp["NUM_EPOCHS"]
        self.base_learning_rate = self.model.hp["LEARNING_RATE"]
        self.momentum = self.model.hp["MOMENTUM"]
        self.decay_step = self.model.hp["DECAY_STEP"]
        self.decay_rate = self.model.hp["DECAY_RATE"]

        self.num_classes = 2

        self.bn_init_decay = 0.5
        self.bn_decay_decay_rate = 0.5
        self.bn_decay_decay_step = float(self.decay_step)
        self.bn_decay_clip = 0.99

        # Saving params
        self.data_dir = resource_filename('caveolae_cls', '/data')
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.data_fn = FLAGS.input_type + '_' + FLAGS.model + '_' + ((self.mil + '_') if self.mil is not None else '') \
                       + time.strftime("%Y-%m-%d_%H:%M")
        self.data_fn = os.path.join(self.data_dir, self.data_fn)
        self.model_name = self.flags.model if FLAGS.model_name is None else FLAGS.model_name
        self.model_save_path = os.path.join(self.data_dir, "saved_models", self.model_name)
        self.step_save_path = os.path.join(self.model_save_path, "step")
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            os.makedirs(self.step_save_path)

        self.metrics = {
            'training_loss': [None], 'validation_loss': [],
            'training_accuracy': [None], 'validation_accuracy': [],
            'training_precision': [None], 'validation_precision': [],
            'training_sensitivity': [None], 'validation_sensitivity': [],
            'training_specificity': [None], 'validation_specificity': [],
            'training_f1': [None], 'validation_f1': []}
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.count = 0
        self.loss = 0.
        self.val_loss = 0.

    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,  # Base learning rate.
            batch * self.batch_size,  # Current index into the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, self.base_learning_rate * 0.001)  # CLIP THE LEARNING RATE!
        return learning_rate # self.base_learning_rate # TODO fix this

    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            self.bn_init_decay,
            batch * self.batch_size,
            self.bn_decay_decay_step,
            self.bn_decay_decay_rate,
            staircase=True)
        bn_decay = tf.minimum(self.bn_decay_clip, 1 - bn_momentum)
        return bn_decay

    def reset_scores(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.count = 0
        self.loss = 0.
        self.val_loss = 0.

    def update_scores(self, loss, true, pred, val_loss=0):
        self.loss += float(loss / float(self.batch_size))
        self.count += 1
        self.val_loss += float(val_loss / float(self.batch_size))
        if self.classification:
            old_sum = self.tp + self.tn + self.fp + self.fn
            self.tp += np.count_nonzero(true * pred)
            self.tn += np.count_nonzero((true - 1) * (pred - 1))
            self.fp += np.count_nonzero((true - 1) * pred)
            self.fn += np.count_nonzero(true * (pred - 1))
            # print self.tp, self.tn, self.fp, self.fn, " == ", old_sum, self.model.hp['BATCH_SIZE']
            assert (old_sum + self.model.hp['BATCH_SIZE'] == self.tp + self.tn + self.fp + self.fn)

    def calculate_metrics(self, reset_scores=True):
        loss = self.loss / float(self.count)
        val_loss = self.val_loss / float(self.count)
        accuracy = 0. if self.tp + self.tn == 0 else (self.tp + self.tn) / float(self.tp + self.tn + self.fp + self.fn)
        precision = 0. if self.tp == 0 else self.tp / float(self.tp + self.fp)
        sensitivity = 0. if self.tp == 0 else self.tp / float(self.tp + self.fn)
        specificity = 0. if self.tn == 0 else self.tn / float(self.tn + self.fp)
        f1 = 0. if precision + sensitivity == 0 else 2 * precision * sensitivity / (precision + sensitivity)
        if reset_scores:
            self.reset_scores()
        return loss, accuracy, precision, sensitivity, specificity, f1, val_loss

    def update_metrics(self, loss, accuracy, precision, sensitivity, specificity, f1, val_loss=0, training=True):
        prefix = 'training_' if training else 'validation_'
        # update
        self.metrics[prefix + 'loss'].append(loss)
        if self.classification:
            self.metrics[prefix + 'accuracy'].append(accuracy)
            self.metrics[prefix + 'precision'].append(precision)
            self.metrics[prefix + 'sensitivity'].append(sensitivity)
            self.metrics[prefix + 'specificity'].append(specificity)
            self.metrics[prefix + 'f1'].append(f1)
        # print
        print prefix + 'loss: ' + str(loss)
        if not training:
            print prefix + 'val_loss: ' + str(val_loss)
        if self.classification:
            print prefix + 'accuracy: ' + str(accuracy)
            print prefix + 'precision: ' + str(precision)
            print prefix + 'sensitivity: ' + str(sensitivity)
            print prefix + 'specificity: ' + str(specificity)
            print prefix + 'f1: ' + str(f1)

    def train(self):
        with tf.Graph().as_default():
            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)
            if self.flags.model == "cae_cnn":
                self.model.data_handler.sess = sess
                self.model.data_handler.model_save_path = os.path.join(self.data_dir, "saved_models", "cae")
            with tf.device('/gpu:' + str(self.gpu_index)):
                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                step = tf.Variable(0)
                bn_decay = self.get_bn_decay(step)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                self.model.generate(bn_decay=bn_decay)

                # Get training operator
                learning_rate = self.get_learning_rate(step)
                tf.summary.scalar('learning_rate', learning_rate)
                if self.optimizer == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                           momentum=self.momentum)
                elif self.optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(self.model.loss, global_step=step)

            # Init variables
            init = tf.global_variables_initializer()

            sess.run(init, {self.model.is_training: True})

            global_step_saver = tf.train.Saver(var_list=[step])

            if self.flags.load_model == "True":
                self.model.restore(sess, self.model_save_path)
                global_step_saver.restore(sess, os.path.join(self.step_save_path))

            ops = {'train_op': train_op, 'step': step}

            print "Initialization evaluation"
            # self.eval_one_epoch(sess, ops, -1)
            print "Instance level TRAINING eval"
            full_model = self.model
            self.model = self.model.model
            self.eval_one_epoch(sess, ops, -1, use_training_data=True)
            self.model = full_model
            print "Instance level VALIDATION eval"
            full_model = self.model
            self.model = self.model.model
            self.eval_one_epoch(sess, ops, -1, use_training_data=False)
            self.model = full_model

            for epoch in range(self.max_epoch):
                print '-' * 10 + ' Epoch: %03d ' % epoch + '-' * 10
                sys.stdout.flush()

                self.train_one_epoch(sess, ops, epoch)

                # eval_ model
                # self.eval_one_epoch(sess, ops, epoch)

                # if mil, eval_ instance model
                if self.mil is not None:
                    print "Instance level TRAINING eval"
                    full_model = self.model
                    self.model = self.model.model
                    self.eval_one_epoch(sess, ops, epoch, use_training_data=True)
                    self.model = full_model
                    print "Instance level VALIDATION eval"
                    full_model = self.model
                    self.model = self.model.model
                    self.eval_one_epoch(sess, ops, epoch, use_training_data=False)
                    self.model = full_model


                # Save the variables to disk.
                if epoch % 10 == 0:
                    self.model.save(sess, self.model_save_path, global_step=step)
                    global_step_saver.save(sess, os.path.join(self.step_save_path, "step"), global_step=step)

        pickle.dump([vars(self.flags), self.model.hp, self.metrics], open(self.data_fn, "wb"))

    def train_one_epoch(self, sess, ops, epoch):
        """ ops: dict mapping from string to tf ops """
        is_training = True
        # Shuffle train files
        for data, labels in self.model.get_batch():
            feed_dict = {self.model.input_pl: data,
                         self.model.is_training: is_training}
            if self.classification:
                feed_dict[self.model.label_pl] = labels
            step, _, loss, pred_val = sess.run(
                [ops['step'], ops['train_op'], self.model.loss, self.model.pred], feed_dict=feed_dict)
            # train_writer.add_summary(summary, step)
            # print "****", mean, "****", len(pos_idx)
            if self.model.use_softmax:
                # print pred_val
                pred_val = np.argmax(pred_val, axis=1)
                labels = np.argmax(labels, axis=1)
            else:
                np.rint(pred_val)
            pred_val = pred_val.flatten()
            # calculate metrics
            self.update_scores(loss, labels, pred_val)

        loss, accuracy, precision, sensitivity, specificity, f1, _ = self.calculate_metrics()
        self.update_metrics(loss, accuracy, precision, sensitivity, specificity, f1, training=True)

    def eval_one_epoch(self, sess, ops, epoch, use_training_data=False):
        """ ops: dict mapping from string to tf ops """
        is_training = False

        ### REMOVE ### TODO
        # n = 0
        # cae_plotting_data = {}
        ##############
        use = 'train' if use_training_data else 'val'
        for data, labels in self.model.get_batch(use=use):
            feed_dict = {self.model.input_pl: data,
                         self.model.is_training: is_training}
            if self.classification:
                feed_dict[self.model.label_pl] = labels
            step, loss, val_loss, pred_val = sess.run([ops['step'], self.model.loss, self.model.val_loss, self.model.pred], feed_dict=feed_dict)

            ### REMOVE ### TODO
            # if n % 10 == 0:
            #     cae_plotting_data[n] = (data[0], pred_val[0])
            # n += 1
            ##############

            if self.model.use_softmax:
                # print pred_val
                pred_val = np.argmax(pred_val, axis=1)
                labels = np.argmax(labels, axis=1)
            else:
                np.rint(pred_val)
            pred_val = pred_val.flatten()
            # calculate metrics
            self.update_scores(loss, labels, pred_val, val_loss=val_loss)

        loss, accuracy, precision, sensitivity, specificity, f1,  val_loss = self.calculate_metrics()
        self.update_metrics(loss, accuracy, precision, sensitivity, specificity, f1, val_loss=val_loss, training=False)

        ### REMOVE ### TODO
        # filename = os.path.join(self.data_dir, str(epoch) + ".p")
        # with open(filename, "wb") as f:
        #     pickle.dump(cae_plotting_data, f)
        ##############


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='pointnet',
                        help='Model type: pointnet, cnn, cae_cnn [default: pointnet]')
    parser.add_argument('--mil', default=None,
                        help='Multiple instance method: seg, sub, or None [default: None]')
    parser.add_argument('--input_type', default='projection',
                        help='pointcloud or projection [default: projection]')
    parser.add_argument('--optimizer', default='adam',
                        help='adam or momentum [default: adam]')
    parser.add_argument('--load_model', default='False',
                        help='True or False [default: False]')
    parser.add_argument('--model_name', default=None,
                        help='Name to save model to [default: model]')

    flags = parser.parse_args()

    t = Train(flags)
    t.train()

if __name__ == "__main__":
    main()