import argparse
import numpy as np
import tensorflow as tf
import os
import sys

import caveolae_cls.pointnet.pointnet_model as pn
import caveolae_cls.cnn.cnn_model as cnn

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet',
                    help='Model name: pointnet [default: pointnet]')
parser.add_argument('--input_type', default='pointcloud',
                    help='Model name: pointcloud, multiview, voxels [default: pointcloud]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')

FLAGS = parser.parse_args()

if FLAGS.model == "pointnet":
    MODEL = pn.PointNet()
elif FLAGS.model == "cnn":
    MODEL = cnn.CNN(FLAGS.input_type)

GPU_INDEX = FLAGS.gpu
OPTIMIZER = FLAGS.optimizer
LOG_DIR = FLAGS.log_dir

BATCH_SIZE = MODEL.hp["BATCH_SIZE"]
NUM_POINT = MODEL.hp["NUM_POINTS"]
MAX_EPOCH = MODEL.hp["NUM_EPOCHS"]
BASE_LEARNING_RATE = MODEL.hp["LEARNING_RATE"]
MOMENTUM = MODEL.hp["MOMENTUM"]
DECAY_STEP = MODEL.hp["DECAY_STEP"]
DECAY_RATE = MODEL.hp["DECAY_RATE"]

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            data_pl, labels_pl = MODEL.get_input_placeholders(BATCH_SIZE)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = MODEL.get_model(data_pl, is_training_pl,
                                               bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(
                BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'data_pl': data_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess,
                                       os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Force number of files to be a multiple of batch size
    batch_shape = [BATCH_SIZE, NUM_POINT, 3]

    # Shuffle train files
    total_correct = 0
    total_seen = 0
    total_positives = 0
    loss_sum = 0
    num_batches = 0

    for data, labels in MODEL.get_batch(batch_shape, 1):
        num_batches += 1
        total_positives += np.sum(labels)
        data = data[:, 0:NUM_POINT, :]  # TODO should be unecessary

        feed_dict = {ops['data_pl']: data,
                     ops['labels_pl']: labels,
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss_val, pred_val = sess.run(
            [ops['merged'], ops['step'],
             ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = pred_val.flatten()
        pred_val = np.rint(pred_val)
        correct = np.sum(pred_val == labels)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

    print "Positive clusters: %d" % total_positives
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # Force number of files to be a multiple of batch size
    batch_shape = [BATCH_SIZE, NUM_POINT, 3]

    # Shuffle train files

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for data, labels in MODEL.get_batch(batch_shape, 1, eval=True):
        data = data[:, 0:NUM_POINT, :]  # TODO should be unecessary
        feed_dict = {ops['data_pl']: data,
                     ops['labels_pl']: labels,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = pred_val.flatten()
        pred_val = np.rint(pred_val)
        correct = np.sum(pred_val == labels)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(BATCH_SIZE):
            l = int(labels[i])
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)
                                                   / np.array(total_seen_class,
                                                              dtype=np.float))))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()