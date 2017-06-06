import tensorflow as tf
import nn_layers

def projection_model(input_data, input_labels):
    """
    Uses 3D point cloud mapped to projection. Size is 600x600 with 3 channels (xy, yz, zx)
    """
    conv1 = nn_layers.conv2d(input_data, 3, 32, (3, 3), 'conv_1')
    conv2 = nn_layers.conv2d(conv1, 32, 64, (3, 3), 'conv_2')
    pool1 = nn_layers.max_pool2d(conv2, (3, 3), 'pool1')
    conv3 = nn_layers.conv2d(pool1, 64, 128, (3, 3), 'conv_3')
    pool2 = nn_layers.max_pool2d(conv3, (3, 3), 'pool2')
    conv4 = nn_layers.conv2d(pool2, 128, 128, (3, 3), 'conv_4')
    pool3 = nn_layers.max_pool2d(conv4, (3, 3), 'pool3')
    conv5 = nn_layers.conv2d(pool3, 128, 256, (3, 3), 'conv_5')
    pool4 = nn_layers.max_pool2d(conv5, (3, 3), 'pool4')
    conv6 = nn_layers.conv2d(pool4, 256, 1024, (1, 1), 'conv_6')
    conv7 = nn_layers.conv2d(conv6, 1024, 2, (1, 1), 'conv_7')
    n_and = nn_layers.noisy_and(conv7, 2, 'noisy_and')
    y_hat = nn_layers.fc_layer(n_and, 2, 1, 'predicted_y', act=tf.nn.sigmoid, 
                               use_batch_norm=False)

    diff = input_labels * tf.log(tf.clip_by_value(y_hat,1e-16,1.0))
    cross_entropy = -tf.reduce_mean(diff)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
