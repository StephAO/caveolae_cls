import tensorflow as tf
import nn_layers

def projection_model(input_data, input_labels):
    """
    Uses 3D point cloud mapped to projection. Size is 600x600 with 3 channels (xy, yz, zx)
    """
    conv1 = nn_layers.conv_layer(input_data, 3, 3, 3, 32, 1, 'conv_1')
    conv2 = nn_layers.conv_layer(conv1, 3, 3, 32, 64, 1, 'conv_2')
    pool1 = nn_layers.pool_layer(conv2, 3, 'pool1')
    conv3 = nn_layers.conv_layer(pool1, 3, 3, 64, 128, 1, 'conv_3')
    pool2 = nn_layers.pool_layer(conv3, 3, 'pool2')
    conv4 = nn_layers.conv_layer(pool2, 3, 3, 128, 128, 1, 'conv_4')
    pool3 = nn_layers.pool_layer(conv4, 3, 'pool3')
    conv5 = nn_layers.conv_layer(pool3, 3, 3, 128, 256, 1, 'conv_5')
    pool4 = nn_layers.pool_layer(conv5, 3, 'pool4')
    conv6 = nn_layers.conv_layer(pool4, 1, 1, 256, 1024, 1, 'conv_6')
    conv7 = nn_layers.conv_layer(conv6, 1, 1, 1024, 2, 1, 'conv_7')
    pool5 = nn_layers.noisy_and(conv7, 2, 'noisy_and')
    y_hat = nn_layers.fc_layer(pool5, 2, 1, 'predicted_y', act=tf.nn.sigmoid, use_batch_norm=False)


    with tf.name_scope('cross_entropy'):
        diff = input_labels * tf.log(tf.clip_by_value(y_hat,1e-16,1.0))
        with tf.name_scope('total'):
            cross_entropy = -tf.reduce_mean(diff)
        tf.summary.scalar('cross entropy', cross_entropy)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)