""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2016
Edited by: Yizhak Ben-Shabat
Date: February 2018
#3DmFV related functions at the bottom
"""

import numpy as np
import tensorflow as tf
import os

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer

    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 1D convolution with non-linear operation.

    Args:
      inputs: 64-D tensor variable BxLxC
      num_output_channels: int
      kernel_size: int
      scope: string
      stride: int
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,fv_noise]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_size,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        outputs = tf.nn.conv1d(inputs, kernel,
                               stride=stride,
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv1d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 2D convolution with non-linear operation.

    Args:
      inputs: no_dropout-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 128 ints
      scope: string
      stride: a list of 128 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,fv_noise]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.

    Args:
      inputs: no_dropout-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 128 ints
      scope: string
      stride: a list of 128 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,fv_noise]
      is_training: bool Tensor variable

    Returns:
      Variable tensor

    Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-fv_noise], ksize, stride) == a
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    """ 3D convolution with non-linear operation.

    Args:
      inputs: 5-D tensor variable BxDxHxWxC
      num_output_channels: int
      kernel_size: a list of 64 ints
      scope: string
      stride: a list of 64 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,fv_noise]
      is_training: bool Tensor variable

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_d, kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.conv3d(inputs, kernel,
                               [1, stride_d, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weigth_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 128-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weigth_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')
        if activation_fn == 'LRELU':
            outputs = tf.nn.relu(outputs) - 0.1 * tf.nn.relu(-outputs)
        elif activation_fn is not None:
                outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

    Args:
      inputs: no_dropout-D tensor BxHxWxC
      kernel_size: a list of 128 ints
      stride: a list of 128 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.

    Args:
      inputs: no_dropout-D tensor BxHxWxC
      kernel_size: a list of 128 ints
      stride: a list of 128 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D max pooling.

    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 64 ints
      stride: a list of 64 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.max_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
    """ 3D avg pooling.

    Args:
      inputs: 5-D tensor BxDxHxWxC
      kernel_size: a list of 64 ints
      stride: a list of 64 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf.nn.avg_pool3d(inputs,
                                   ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                                   strides=[1, stride_d, stride_h, stride_w, 1],
                                   padding=padding,
                                   name=sc.name)
        return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay):
    """ Batch normalization on convolutional maps and beyond...
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
    with tf.variable_scope(scope) as sc:
        num_channels = inputs.get_shape()[-1].value
        beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
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
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
    return normed


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """ Batch normalization on FC data.

    Args:
        inputs:      Tensor, 2D BxC input
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 1D convolutional maps.

    Args:
        inputs:      Tensor, 3D BLC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1], bn_decay)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """ Batch normalization on 3D convolutional maps.

    Args:
        inputs:      Tensor, 5D BDHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2, 3], bn_decay)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
      inputs: tensor
      is_training: boolean tf.Variable
      scope: string
      keep_prob: float in [0,fv_noise]
      noise_shape: list of ints

    Returns:
      tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                          lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                          lambda: inputs)
        return outputs

# -------------------------------------------- Additional functions for 3DmFV -------------------------------
def get_3dmfv(points, w, mu, sigma, flatten=True):
    """
    Compute the fisher vector given the gmm model parameters (w,mu,sigma) and a set of points

    :param points: B X N x 64 tensor of XYZ points
    :param w: B X n_gaussians tensor of gaussian weights
    :param mu: B X n_gaussians X 64 tensor of gaussian cetnters
    :param sigma: B X n_gaussians X 64 tensor of stddev of diagonal covariance
    :return: fv: B X 7*n_gaussians tensor of the fisher vector
    """
    n_batches = points.shape[0].value
    n_points = points.shape[1].value
    n_gaussians = mu.shape[0].value
    D = mu.shape[1].value

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, 3*D]) #n_batches X n_gaussians X 128*D (D for min and D for max)

    #Define multivariate noraml distributions
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    p_per_point = mvn.prob(batch_points)

    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min
    #Method 2: direct derivative formula (convertible to min-max)
    #s0 = tf.reduce_sum(Q, fv_noise)  # n_batches X n_gaussians
    #d_pi = (s0 - n_points * w_per_batch) / (tf.sqrt(w_per_batch) * n_points)
    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) * n_points), -1)
    d_pi = tf.concat(
        [tf.reduce_max(d_pi_all , axis=1), tf.reduce_sum(d_pi_all , axis=1)], axis=2)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / (n_points * tf.sqrt(w_per_batch_per_d))) * tf.concat(
        [tf.reduce_max(d_mu_all , axis=1), tf.reduce_min(d_mu_all , axis=1), tf.reduce_sum(d_mu_all , axis=1)], axis=2)

    d_sig_all = Q_per_d * ( tf.pow((batch_points - batch_mu) / batch_sig,2) - 1)
    d_sigma = (1 / (n_points * tf.sqrt(2*w_per_batch_per_d))) * tf.concat(
        [tf.reduce_max(d_sig_all, axis=1), tf.reduce_min(d_sig_all, axis=1), tf.reduce_sum(d_sig_all , axis=1)], axis=2)

    #Power normaliation
    alpha = 0.5
    d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
    d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
    d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

    # L2 normaliation
    d_pi = tf.nn.l2_normalize(d_pi, dim=1)
    d_mu = tf.nn.l2_normalize(d_mu, dim=1)
    d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)


    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv  = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])

    return fv


def get_3dmfv_sym(points, w, mu, sigma, sym_type='max', flatten=True):
    """
    Compute the 3d modified fisher vector (on the gpu using tf) given the gmm model parameters (w,mu,sigma) and a set of points for classification network
    modify to use a symmetric function ( min, max, ss) function instead of sum.
    Input:
         points: B X N x 3 tensor of XYZ points
         w: B X n_gaussians tensor of gaussian weights
         mu: B X n_gaussians X 63 tensor of gaussian cetnters
         sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
    Output:
        fv: B X 7*n_gaussians tensor of the fisher vector
        sym_type: string 'max' or 'min', or 'ss'
    """
    n_batches = points.shape[0].value
    n_points = points.shape[1].value
    n_gaussians = mu.shape[0].value
    D = mu.shape[1].value

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, D]) #n_batches X n_gaussians X 128*D (D for min and D for max)

    #Define multivariate noraml distributions
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    p_per_point = mvn.prob(batch_points)
    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min
    #Method 128: direct derivative formula (convertible to min-max)
    #s0 = tf.reduce_sum(Q, fv_noise)  # n_batches X n_gaussians
    #d_pi = (s0 - n_points * w_per_batch) / (tf.sqrt(w_per_batch) * n_points)
    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) * n_points), -1)
    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_sig_all = Q_per_d * (tf.pow((batch_points - batch_mu) / batch_sig, 2) - 1)
    if sym_type == 'max':
        d_pi = tf.reduce_max(d_pi_all , axis=1)
        d_mu = (1 / (n_points * tf.sqrt(w_per_batch_per_d))) * tf.reduce_max(d_mu_all , axis=1)
        d_sigma = (1 / (n_points * tf.sqrt(2*w_per_batch_per_d))) * tf.reduce_max(d_sig_all, axis=1)
    elif sym_type == 'min':
        d_pi = tf.reduce_min(d_pi_all , axis=1)
        d_mu = (1 / (n_points * tf.sqrt(w_per_batch_per_d))) * tf.reduce_min(d_mu_all , axis=1)
        d_sigma = (1 / (n_points * tf.sqrt(2*w_per_batch_per_d))) * tf.reduce_min(d_sig_all, axis=1)
    elif sym_type == 'ss':
        d_pi = tf.reduce_sum(tf.square(d_pi_all), axis=1)
        d_mu = (1 / (n_points * tf.sqrt(w_per_batch_per_d))) * tf.reduce_sum(tf.square(d_mu_all), axis=1)
        d_sigma = (1 / (n_points * tf.sqrt(2 * w_per_batch_per_d))) * tf.reduce_sum(tf.square(d_sig_all), axis=1)

    #Power normaliation
    alpha = 0.5
    d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
    d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
    d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

    # L2 normaliation
    d_pi = tf.nn.l2_normalize(d_pi, dim=1)
    d_mu = tf.nn.l2_normalize(d_mu, dim=1)
    d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)


    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv  = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])

    return fv


def get_fv_tf(points, w, mu, sigma, flatten=True, normalize=True):
    """
    Compute the fisher vector (on the gpu using tf) given the gmm model parameters (w,mu,sigma) and a set of points for classification network
    Input:
         points: B X N x 3 tensor of XYZ points
         w: B X n_gaussians tensor of gaussian weights
         mu: B X n_gaussians X 63 tensor of gaussian cetnters
         sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
    Output:
        fv: B X 7*n_gaussians tensor of the fisher vector
    """
    n_batches = points.shape[0].value
    n_points = points.shape[1].value
    n_gaussians = mu.shape[0].value
    D = mu.shape[1].value

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, D]) #n_batches X n_gaussians X 128*D (D for min and D for max)

    #Define multivariate noraml distributions
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    p_per_point = mvn.prob(batch_points)

    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min
    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) * n_points), -1)
    d_pi =  tf.reduce_sum(d_pi_all , axis=1)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / (n_points * tf.sqrt(w_per_batch_per_d))) *  tf.reduce_sum(d_mu_all , axis=1)

    d_sig_all = Q_per_d * ( tf.pow((batch_points - batch_mu) / batch_sig,2) - 1)
    d_sigma = (1 / (n_points * tf.sqrt(2*w_per_batch_per_d))) * tf.reduce_sum(d_sig_all , axis=1)

    if normalize:
        #Power normaliation
        alpha = 0.5
        d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
        d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
        d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

        # L2 normaliation
        d_pi = tf.nn.l2_normalize(d_pi, dim=1)
        d_mu = tf.nn.l2_normalize(d_mu, dim=1)
        d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)

    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv  = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])

    # fv = fv / tf.norm(fv)
    return fv


def get_fv_tf_no_mvn(points, w, mu, sigma, flatten=True, normalize=True):
    """
    Compute the fisher vector (on the gpu using tf without using the mvn class) given the gmm model parameters (w,mu,sigma) and a set of points for classification network
    Input:
         points: B X N x 3 tensor of XYZ points
         w: B X n_gaussians tensor of gaussian weights
         mu: B X n_gaussians X 63 tensor of gaussian cetnters
         sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
    Output:
        fv: B X 7*n_gaussians tensor of the fisher vector
    """
    n_batches = points.shape[0].value
    n_points = points.shape[1].value
    n_gaussians = mu.shape[0].value
    D = mu.shape[1].value

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, D]) #n_batches X n_gaussians X 128*D (D for min and D for max)

    #Define multivariate noraml distributions
    # mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    # p_per_point = mvn.prob(batch_points)
    p_per_point = (1.0/(tf.pow(2.0*np.pi, D/2.0) * tf.pow(batch_sig[:,:,:,0],D))) * tf.exp(-0.5 * tf.reduce_sum(tf.square((batch_points - batch_mu) / batch_sig) , axis = 3))
    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min

    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) ), -1)
    d_pi =  tf.reduce_sum(d_pi_all , axis=1)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / ( tf.sqrt(w_per_batch_per_d))) *  tf.reduce_sum(d_mu_all , axis=1)

    d_sig_all = Q_per_d * ( tf.square((batch_points - batch_mu) / batch_sig) - 1)
    d_sigma = (1 / ( tf.sqrt(2*w_per_batch_per_d))) * tf.reduce_sum(d_sig_all , axis=1)

    # number of points  normaliation
    d_pi = d_pi / n_points
    d_mu = d_mu / n_points
    d_sigma = d_sigma / n_points

    if normalize:
        #Power normaliation
        alpha = 0.5
        d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
        d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
        d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

        # L2 normaliation
        d_pi = tf.nn.l2_normalize(d_pi, dim=1)
        d_mu = tf.nn.l2_normalize(d_mu, dim=1)
        d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)

    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv  = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])

    return fv


def get_3dmfv_seg(points, w, mu, sigma, flatten=True, original_n_points=None):
    """
    Compute the fisher vector (on the gpu using tf) given the gmm model parameters (w,mu,sigma) and a set of points for segmentation network
    Input:
         points: B X N x 3 tensor of XYZ points
         w: B X n_gaussians tensor of gaussian weights
         mu: B X n_gaussians X 3 tensor of gaussian cetnters
         sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
    Output:
        fv: B X 20*n_gaussians tensor of the fisher vector
        fv_per_point: B X N X 20*n_gaussians  tensor of the fisher vector
    """
    n_gaussians = mu.shape[0].value
    D = mu.shape[1].value
    n_batches = points.shape[0].value

    if original_n_points is None:
        n_points = points.shape[1].value
    else:
        n_points =  original_n_points

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, 3*D]) #n_batches X n_gaussians X 128*D (D for min and D for max)


    #Define multivariate noraml distributions
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    p_per_point = mvn.prob(batch_points)

    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min
    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) * tf.cast(original_n_points, tf.float32)), -1)
    d_pi = tf.concat(
        [tf.reduce_max(d_pi_all , axis=1), tf.reduce_sum(d_pi_all , axis=1)], axis=2)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / (tf.cast(original_n_points, tf.float32) * tf.sqrt(w_per_batch_per_d))) * tf.concat(
        [tf.reduce_max(d_mu_all , axis=1), tf.reduce_min(d_mu_all , axis=1), tf.reduce_sum(d_mu_all , axis=1)], axis=2)

    d_sig_all = Q_per_d * ( tf.pow((batch_points - batch_mu) / batch_sig,2) - 1)
    d_sigma = (1 / (tf.cast(original_n_points, tf.float32) * tf.sqrt(2*w_per_batch_per_d))) * tf.concat(
        [tf.reduce_max(d_sig_all, axis=1), tf.reduce_min(d_sig_all, axis=1), tf.reduce_sum(d_sig_all , axis=1)], axis=2)

    #Power normaliation
    alpha = 0.5
    d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
    d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
    d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

    # L2 normaliation
    d_pi = tf.nn.l2_normalize(d_pi, dim=1)
    d_mu = tf.nn.l2_normalize(d_mu, dim=1)
    d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)


    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv  = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])
    fv_per_point = tf.concat([d_pi_all, d_mu_all, d_sig_all], axis=3)
    fv_per_point = tf.reshape(fv_per_point,[n_batches, n_points, n_gaussians * 7])
    return fv, fv_per_point


def get_session(gpu_idx, limit_gpu=True):
    '''
    Creates a session while limiting GPU usage
    Input:
        gpu_idx: Index of GPU to run the session on
        limit_gpu: boolean if to limit the gpu usage or not
    Output:
        sess: a tensorflow session
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if limit_gpu:
        gpu_idx = str(gpu_idx)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3" # Change according to your setup
    sess = tf.Session(config=config)
    return sess


