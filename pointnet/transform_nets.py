import numpy as np
import tensorflow as tf
import caveolae_cls.nn_layers as nn_layers


def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    input_channels = input_image.get_shape()[-1].value
    net = nn_layers.conv2d(input_image, input_channels, 64, [1, 3],
                           padding='VALID', stride=[1, 1],
                           batch_norm=True, is_training=is_training,
                           layer_name='tconv1', batch_norm_decay=bn_decay)
    net = nn_layers.conv2d(net, 64, 128, [1, 1],
                           padding='VALID', stride=[1, 1],
                           batch_norm=True, is_training=is_training,
                           layer_name='tconv2', batch_norm_decay=bn_decay)
    net = nn_layers.conv2d(net, 128, 1024, [1, 1],
                           padding='VALID', stride=[1, 1],
                           batch_norm=True, is_training=is_training,
                           layer_name='tconv3', batch_norm_decay=bn_decay)
    net = nn_layers.max_pool2d(net, [num_point, 1],
                               padding='VALID', layer_name='layer_name')

    net = tf.reshape(net, [batch_size, -1])
    input_channels = net.get_shape()[-1].value
    net = nn_layers.fc(net, input_channels, 512, batch_norm=True,
                       is_training=is_training, layer_name='tfc1',
                       batch_norm_decay=bn_decay)
    net = nn_layers.fc(net, 512, 256, batch_norm=True, is_training=is_training,
                       layer_name='tfc2', batch_norm_decay=bn_decay)

    with tf.variable_scope('transform_XYZ') as sc:
        assert(K == 3)
        weights = tf.get_variable('weights', [256, 3*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [3*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, 3, K])
    return transform


def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0].value
    num_point = inputs.get_shape()[1].value

    input_channels = inputs.get_shape()[-1].value

    net = nn_layers.conv2d(inputs, input_channels, 64, [1, 64],
                           padding='VALID', stride=[1, 1],
                           batch_norm=True, is_training=is_training,
                           layer_name='fconv1', batch_norm_decay=bn_decay)
    net = nn_layers.conv2d(net, 64, 128, [1, 1],
                           padding='VALID', stride=[1, 1],
                           batch_norm=True, is_training=is_training,
                           layer_name='fconv2', batch_norm_decay=bn_decay)
    net = nn_layers.conv2d(net, 128, 1024, [1, 1],
                           padding='VALID', stride=[1, 1],
                           batch_norm=True, is_training=is_training,
                           layer_name='fconv3', batch_norm_decay=bn_decay)
    net = nn_layers.max_pool2d(net, [num_point, 1],
                               padding='VALID', layer_name='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    input_channels = net.get_shape()[-1].value
    net = nn_layers.fc(net, input_channels, 512, batch_norm=True,
                       is_training=is_training, layer_name='tfc1',
                       batch_norm_decay=bn_decay)
    net = nn_layers.fc(net, 512, 256, batch_norm=True,
                       is_training=is_training, layer_name='tfc2',
                       batch_norm_decay=bn_decay)

    with tf.variable_scope('transform_feat') as sc:
        weights = tf.get_variable('weights', [256, K*K],
                                  initializer=tf.constant_initializer(0.0),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', [K*K],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform
