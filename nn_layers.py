import tensorflow as tf


def weight_variable(name, shape, use_xavier=True, wd=None, stddev=1e-3):
    """
    Create an initialized weight Variable with weight decay.
    Args:
        name[string]: name of the variable
        shape[list of ints]: shape of variable
        use_xavier[bool]: whether to use xavier initializer
    Returns:
        Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def bias_variable(name, shape, value=0.1):
    """
    Create an initialized bias variable.
    Args:
        name[string]: name of the variable
        shape[list of ints]: shape of variable
        value[bool]: initial value of bias variable
    Returns:
        Variable Tensor
    """
    initial = tf.constant(value, shape=shape)
    return tf.get_variable(name, initializer=initial)


def variable_summaries(var, name):
    """
    Attach useful summaries to a Tensor.
    Args:
        var[Tensor]: Variable to attach summaries about
        name[string]: name of variable
    """
    with tf.variable_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.variable_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def fc(input_tensor, input_dim, output_dim, layer_name,
       activation_fn=tf.nn.relu, is_training=True, batch_norm=False,
        batch_norm_decay=None, reuse=None):
    """
    Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.variable_scope(layer_name, reuse=reuse):
        weights = weight_variable('weights', [input_dim, output_dim])
        biases = bias_variable('biases', [output_dim])
        output = tf.matmul(input_tensor, weights) + biases
        if batch_norm:
                output = batch_norm_fc(output, is_training=is_training,
                                       bn_decay=batch_norm_decay,
                                       scope=layer_name+'_batch_norm',
                                       reuse=reuse)
        if activation_fn is not None:
                output = activation_fn(output, name='activation')
        return output


def conv2d(input_tensor, num_in_feat_maps, num_out_feat_maps, kernel_size,
           layer_name, stride=[1, 1], padding='SAME', use_xavier=True,
           stddev=1e-3, activation_fn=tf.nn.relu, batch_norm=False,
           batch_norm_decay=None, is_training=None, reuse=None):
    """
    2D convolution with non-linear operation.
    Args:
        input_tensor: 4-D tensor variable BxHxWxC
        num_in_feat_maps: int
        num_out_feat_maps: int
        kernel_size: a list of 2 ints
        layer_name: string used to scope variables in layer
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        activation_fn: function
        batch_norm: bool, whether to use batch norm
        batch_norm_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(layer_name, reuse=reuse) as sc:
        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w, num_in_feat_maps, num_out_feat_maps]
        weights = weight_variable('weights', shape=kernel_shape,
                                  use_xavier=use_xavier, stddev=stddev)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(input_tensor, weights,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = bias_variable('biases', [num_out_feat_maps], value=1e-3)
        outputs = tf.nn.bias_add(outputs, biases)

        if batch_norm:
            outputs = batch_norm_conv2d(outputs, is_training, reuse=reuse,
                                        bn_decay=batch_norm_decay, scope='bn')
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv2d_transpose(input_tensor, num_in_feat_maps, num_out_feat_maps, kernel_size,
           layer_name, stride=[1, 1], padding='SAME', use_xavier=True,
           stddev=1e-3, activation_fn=tf.nn.relu, batch_norm=False,
           batch_norm_decay=None, is_training=None, reuse=None):
    """
    2D convolution with non-linear operation.
    Args:
        input_tensor: 4-D tensor variable BxHxWxC
        num_in_feat_maps: int
        num_out_feat_maps: int
        kernel_size: a list of 2 ints
        layer_name: string used to scope variables in layer
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        activation_fn: function
        batch_norm: bool, whether to use batch norm
        batch_norm_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor

    """
    with tf.variable_scope(layer_name, reuse=reuse) as sc:
        kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_h, kernel_w, num_in_feat_maps, num_out_feat_maps]
        weights = weight_variable('weights', shape=kernel_shape,
                                  use_xavier=use_xavier, stddev=stddev)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d_transpose(input_tensor, weights,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = bias_variable('biases', [num_out_feat_maps], value=1e-3)
        outputs = tf.nn.bias_add(outputs, biases)

        if batch_norm:
            outputs = batch_norm_conv2d(outputs, is_training, reuse=reuse,
                                        bn_decay=batch_norm_decay, scope='bn')
        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs


def conv3d(input_tensor, num_in_feat_maps, num_out_feat_maps, kernel_size,
           layer_name, stride=[1, 1, 1], padding='SAME', use_xavier=True,
           stddev=1e-3, activation_fn=tf.nn.relu, batch_norm=False,
           batch_norm_decay=None, is_training=None, reuse=None):
    """ 
    3D convolution with non-linear operation.
    Args:
        input_tensor: 5-D tensor variable BxDxHxWxC
        num_in_feat_maps: int
        num_out_feat_maps: int
        kernel_size: a list of 3 ints
        layer_name: string used to scope variables in layer
        stride: a list of 3 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        activation_fn: function
        batch_norm: bool, whether to use batch norm
        batch_norm_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(layer_name, reuse=reuse) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_feat_maps, num_out_feat_maps]
        kernel = weight_variable('weights', shape=kernel_shape,
                                 use_xavier=use_xavier, stddev=stddev)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(input_tensor, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = bias_variable('biases', [num_out_feat_maps],
                               tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if batch_norm:
            outputs = batch_norm_conv3d(outputs, is_training, reuse=reuse,
                                        bn_decay=batch_norm_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(input_tensor, kernel_size, layer_name, stride=[2, 2],
               padding='VALID', reuse=None):
    """
    2D max pooling.
    Args:
        input_tensor: 4-D tensor BxHxWxC
        kernel_size: a list of 2 ints
        layer_name: string to scope variables
        stride: a list of 2 ints
        padding: string, either 'VALID' or 'SAME'
        reuse: Bool, reuse variable in scope, used for mil

    Returns:
        Variable tensor
    """
    with tf.variable_scope(layer_name, reuse=reuse) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(input_tensor,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def max_pool3d(inputs, kernel_size, layer_name, stride=[2, 2, 2],
               padding='VALID', reuse=None):
    """
    3D max pooling.
    Args:
        inputs: 5-D tensor BxDxHxWxC
        kernel_size: a list of 3 ints
        layer_name: string to scope variables
        stride: a list of 3 ints
        padding: string, either 'VALID' or 'SAME'

    Returns:
        Variable tensor
    """
    with tf.variable_scope(layer_name, reuse=reuse) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def noisy_and(input_tensor, num_classes, layer_name="noisy_and", reuse=None):
    """
    @author: Dan Salo, Nov 2016
    Multiple Instance Learning (MIL), flexible pooling function
    Args:
        input_tensor: 4-D tensor BxHxWxC
        num_classes: int, determine number of output maps
        layer_name: string to scope variables

    Return:
        threshold: transformed tensor
    """
    assert input_tensor.get_shape()[3] == num_classes

    with tf.variable_scope(layer_name, reuse=reuse):
        a = bias_variable([1])
        b = bias_variable([1, num_classes], value=0.0)
        mean = tf.reduce_mean(input_tensor, axis=[1, 2])
        threshold = (tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)) / \
                    (tf.sigmoid(a * (1 - b)) - tf.sigmoid(-a * b))
    return threshold


def noisy_and_1d(input_tensor, num_classes, layer_name="noisy_and", reuse=None):
    """
    @author: Dan Salo, Nov 2016
    Multiple Instance Learning (MIL), flexible pooling function
    Args:
        input_tensor: 2-D tensor BxC
        num_classes: int, determine number of output maps
        layer_name: string to scope variables

    Return:
        threshold: transformed tensor
    """
    assert input_tensor.get_shape()[2] == num_classes

    with tf.variable_scope(layer_name, reuse=reuse):
        a = bias_variable('a', [1])
        b = bias_variable('b', [1, num_classes], value=0.0)
        mean = tf.reduce_mean(input_tensor, axis=[1])
        threshold = (tf.nn.sigmoid(a * (mean - b)) - tf.nn.sigmoid(-a * b)) / \
                    (tf.sigmoid(a * (1 - b)) - tf.sigmoid(-a * b))
    return threshold


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay,
                        reuse=None):
    """ 
    Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
        inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        is_training:   boolean tf.Varialbe, true indicates training phase
        scope:         string, variable scope
        moments_dims:  a list of ints, indicating dimensions for moments calculation
        bn_decay:      float or float tensor variable, controling moving average weight
    Return:
        normed:        batch-normalized maps
    """
    with tf.variable_scope(scope, reuse=reuse) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.get_variable('beta',
                               initializer=tf.constant(0.0,
                                                       shape=[num_channels]))
        gamma = tf.get_variable('gamma',
                                initializer=tf.constant(1.0,
                                                        shape=[num_channels]))
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims,
                                              name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay, name='ema')
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (
                            ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_fc(inputs, is_training, bn_decay, scope, reuse=None):
    """ 
    Batch normalization on FC data.
    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay,
                               reuse=reuse)


def batch_norm_conv2d(inputs, is_training, bn_decay, scope, reuse=None):
    """ 
    Batch normalization on 2D convolutional maps.
    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay,
                               reuse=reuse)


def batch_norm_conv3d(inputs, is_training, bn_decay, scope, reuse=None):
    """ 
    Batch normalization on 3D convolutional maps.
    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3],
                               bn_decay, reuse=reuse)


def dropout(inputs,
            is_training,
            layer_name,
            keep_prob=0.5,
            noise_shape=None,):
    """ 
    Dropout layer.
    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      layer_name: string
      keep_prob: float in [0,1]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.variable_scope(layer_name) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs