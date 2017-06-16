import argparse
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time
from pkg_resources import resource_filename

import caveolae_cls.pointnet.pointnet_model as pn
import caveolae_cls.cnn.cnn_model as cnn
import caveolae_cls.segmented_mil.segmented_mil_model as segmented_mil
import caveolae_cls.subregion_mil.subregion_mil_model as subregion_mil


class Train:

    def __init__(self, FLAGS):
        self.flags = FLAGS
        # Select model
        if FLAGS.model == "pointnet":
            self.model = pn.PointNet()
            FLAGS.input_type = "pointcloud"
        elif FLAGS.model == "cnn":
            self.model = cnn.CNN(FLAGS.input_type)
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

        self.data_dir = resource_filename('caveolae_cls', '/data')
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.data_fn = FLAGS.input_type + '_' + FLAGS.model + '_' + ((self.mil + '_') if self.mil is not None else '') \
                 + time.strftime("%Y-%m-%d_%H:%M")
        self.data_fn =  os.path.join(self.data_dir, self.data_fn)


    def get_learning_rate(self, batch):
        learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,  # Base learning rate.
            batch * self.batch_size,  # Current index into the dataset.
            self.decay_step,  # Decay step.
            self.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
        return learning_rate


    def get_bn_decay(self, batch):
        bn_momentum = tf.train.exponential_decay(
            self.bn_init_decay,
            batch * self.batch_size,
            self.bn_decay_decay_step,
            self.bn_decay_decay_rate,
            staircase=True)
        bn_decay = tf.minimum(self.bn_decay_clip, 1 - bn_momentum)
        return bn_decay


    def train(self):
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(self.gpu_index)):
                # Note the global_step=batch parameter to minimize.
                # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
                batch = tf.Variable(0)
                bn_decay = self.get_bn_decay(batch)
                tf.summary.scalar('bn_decay', bn_decay)

                # Get model and loss
                self.model.generate(bn_decay=bn_decay)
                # tf.summary.scalar('loss', loss)

                # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(
                #     self.batch_size)
                # tf.summary.scalar('accuracy', accuracy)

                # Get training operator
                learning_rate = self.get_learning_rate(batch)
                tf.summary.scalar('learning_rate', learning_rate)
                if self.optimizer == 'momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                           momentum=self.momentum)
                elif self.optimizer == 'adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate)
                train_op = optimizer.minimize(self.model.loss, global_step=batch)

                # Add ops to save and restore all the variables.
                # saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            config.log_device_placement = False
            sess = tf.Session(config=config)

            # Add summary writers
            # merged = tf.merge_all_summaries()
            # merged = tf.summary.merge_all()
            # train_writer = tf.summary.FileWriter(os.path.join(self.data_dir, 'train'),
            #                                      sess.graph)
            # test_writer = tf.summary.FileWriter(os.path.join(self.data_dir, 'test'))

            # Init variables
            init = tf.global_variables_initializer()
            # To fix the bug introduced in TF 0.12.1 as in
            # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
            # sess.run(init)
            sess.run(init, {self.model.is_training: True})

            ops = {'train_op': train_op, 'step': batch}

            metrics = {'t_loss': [None], 'v_loss': [], 't_acc': [None], 'v_acc': []}

            print "No training eval"
            self.eval_one_epoch(sess, ops, metrics)

            for epoch in range(self.max_epoch):
                print '**** EPOCH %03d ****' % (epoch)
                sys.stdout.flush()

                self.train_one_epoch(sess, ops, metrics)
                self.eval_one_epoch(sess, ops, metrics)

                # Save the variables to disk.
                # if epoch % 10 == 0:
                #     save_path = saver.save(sess,
                #                            os.path.join(self.data_dir, "model.ckpt"))
                #     log_string("Model saved in file: %s" % save_path)

        pickle.dump([vars(self.flags), self.model.hp, metrics], open(self.data_fn, "wb"))

    def train_one_epoch(self, sess, ops, metrics):
        """ ops: dict mapping from string to tf ops """
        is_training = True

        # Force number of files to be a multiple of batch size
        # batch_shape = [self.batch_size, NUM_POINT, 3]

        # Shuffle train files
        total_correct = 0
        total_seen = 0
        total_positives = 0
        loss_sum = 0
        num_batches = 0

        for data, labels in self.model.get_batch():
            num_batches += 1
            total_positives += np.sum(labels)
            feed_dict = {self.model.input_pl: data,
                         self.model.label_pl: labels,
                         self.model.is_training: is_training}
            step, _, loss_val, pred_val = sess.run(
                [ops['step'], ops['train_op'], self.model.loss, self.model.pred], feed_dict=feed_dict)
            # train_writer.add_summary(summary, step)
            pred_val = pred_val.flatten()
            pred_val = np.rint(pred_val)
            correct = np.sum(pred_val == labels)
            total_correct += correct
            total_seen += self.batch_size
            loss_sum += loss_val

        print "Positive clusters: %d" % total_positives
        print 'mean loss: %f' % (loss_sum / float(total_seen))
        metrics['t_loss'].append(loss_sum / float(total_seen))
        print 'accuracy: %f' % (total_correct / float(total_seen))
        metrics['t_acc'].append(total_correct / float(total_seen))


    def eval_one_epoch(self, sess, ops, metrics):
        """ ops: dict mapping from string to tf ops """
        if self.mil is not None:
            full_model = self.model
            self.model = self.model.model
        is_training = False
        total_seen_class = [0 for _ in range(self.num_classes)]
        total_correct_class = [0 for _ in range(self.num_classes)]

        # Force number of files to be a multiple of batch size
        # batch_shape = [self.batch_size, NUM_POINT, 3]

        # Shuffle train files

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for data, labels in self.model.get_batch(eval=True):
            feed_dict = {self.model.input_pl: data,
                         self.model.label_pl: labels,
                         self.model.is_training: is_training}
            step, loss_val, pred_val = sess.run([ops['step'], self.model.loss, self.model.pred], feed_dict=feed_dict)
            pred_val = pred_val.flatten()
            pred_val = np.rint(pred_val)
            correct = np.sum(pred_val == labels)
            total_correct += correct
            total_seen += self.batch_size
            loss_sum += loss_val
            for i in range(self.batch_size):
                l = int(labels[i])
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i] == l)

        print 'eval mean loss: %f' % (loss_sum / float(total_seen))
        metrics['v_loss'].append(loss_sum / float(total_seen))
        print 'eval accuracy: %f'% (total_correct / float(total_seen))
        metrics['v_acc'].append(total_correct / float(total_seen))
        print 'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)
                                                       / np.array(total_seen_class,
                                                                  dtype=np.float)))
        if self.mil is not None:
            self.model = full_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')
    parser.add_argument('--model', default='pointnet',
                        help='Model name: pointnet [default: pointnet]')
    parser.add_argument('--mil', default='None',
                        help='Type of self.mil name: seg, sub, or None [default: None]')
    parser.add_argument('--input_type', default='projection',
                        help='Model name: pointcloud, multiview, voxels [default: pointcloud]')
    parser.add_argument('--optimizer', default='adam',
                        help='adam or momentum [default: adam]')

    FLAGS = parser.parse_args()

    t = Train(FLAGS).train()

if __name__ == "__main__":
    main()