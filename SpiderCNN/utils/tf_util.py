import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import numpy as np
import tensorflow as tf



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
           is_training=None,
           is_multi_GPU=False,
           gn=False,
           G=32):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
    is_multi_GPU: bool, whether to use multi GPU

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
        if is_multi_GPU:
          outputs = batch_norm_template_multiGPU(outputs, is_training,
                                                 'bn', [0,1,2], bn_decay)
        else:
          outputs = batch_norm_template(outputs, is_training, 
                                        'bn', [0,1,2], bn_decay)
      if gn:
        outputs = group_norm_for_conv(outputs, G=G, scope='gn')
      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs

def spiderConv(feat,
              idx,
              delta,
              num_conv,
              taylor_channel,
              bn=False,
              is_training=None,
              bn_decay=None,
              gn=False,
              G=32,
              is_multi_GPU=False,
              activation_fn=tf.nn.relu,
              scope='taylor'):
  """ 2D convolution with non-linear operation.

  Args:
    feat: 3-D tensor variable BxNxC
    idx: 3-D tensor variable BxNxk
    delta: 4-D tensor variable BxNxkx3
    num_conv: int
    taylor_channel: int    
    bn: bool, whether to use batch norm
    is_training: bool Tensor variable
    bn_decay: float or float tensor variable in [0,1]
    gn: bool, whether to use group norm
    G: int
    is_multi_GPU: bool, whether to use multi GPU
    activation_fn: function
    scope: string
    

  Returns:
    feat: 3-D tensor variable BxNxC
  """
  with tf.variable_scope(scope) as sc:
      grouped_points = group_point(feat, idx)

      batch_size = grouped_points.get_shape()[0].value
      num_point = grouped_points.get_shape()[1].value
      K_knn = grouped_points.get_shape()[2].value
      in_channels = grouped_points.get_shape()[3].value
      shape = [1, 1, 1, taylor_channel]

      X = delta[:, :, :, 0]
      Y = delta[:, :, :, 1]
      Z = delta[:, :, :, 2]

      X = tf.expand_dims(X, -1)
      Y = tf.expand_dims(Y, -1)
      Z = tf.expand_dims(Z, -1)

      #initialize
      initializer = tf.contrib.layers.xavier_initializer()
      
      w_x = tf.tile(_variable_on_cpu('weight_x', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_y = tf.tile(_variable_on_cpu('weight_y', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_z = tf.tile(_variable_on_cpu('weight_z', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_xyz = tf.tile(_variable_on_cpu('weight_xyz', shape, initializer), [batch_size, num_point, K_knn, 1])
      
      w_xy = tf.tile(_variable_on_cpu('weight_xy', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_yz = tf.tile(_variable_on_cpu('weight_yz', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_xz = tf.tile(_variable_on_cpu('weight_xz', shape, initializer), [batch_size, num_point, K_knn, 1])
      biases = tf.tile(_variable_on_cpu('biases', shape, 
                        tf.constant_initializer(0.0)), [batch_size, num_point, K_knn, 1])
      
      w_xx = tf.tile(_variable_on_cpu('weight_xx', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_yy = tf.tile(_variable_on_cpu('weight_yy', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_zz = tf.tile(_variable_on_cpu('weight_zz', shape, initializer), [batch_size, num_point, K_knn, 1])

      w_xxy = tf.tile(_variable_on_cpu('weight_xxy', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_xyy = tf.tile(_variable_on_cpu('weight_xyy', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_xxz = tf.tile(_variable_on_cpu('weight_xxz', shape, initializer), [batch_size, num_point, K_knn, 1])

      w_xzz = tf.tile(_variable_on_cpu('weight_xzz', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_yyz = tf.tile(_variable_on_cpu('weight_yyz', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_yzz = tf.tile(_variable_on_cpu('weight_yzz', shape, initializer), [batch_size, num_point, K_knn, 1])

      
      w_xxx = tf.tile(_variable_on_cpu('weight_xxx', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_yyy = tf.tile(_variable_on_cpu('weight_yyy', shape, initializer), [batch_size, num_point, K_knn, 1])
      w_zzz = tf.tile(_variable_on_cpu('weight_zzz', shape, initializer), [batch_size, num_point, K_knn, 1])

      
      g1 = w_x * X + w_y * Y + w_z * Z + w_xyz * X * Y * Z
      g2 = w_xy * X * Y + w_yz * Y * Z + w_xz * X * Z + biases
      g3 = w_xx * X * X + w_yy * Y * Y + w_zz * Z * Z
      g4 = w_xxy * X * X * Y + w_xyy * X * Y * Y + w_xxz * X * X * Z
      g5 = w_xzz * X * Z * Z + w_yyz * Y * Y * Z + w_yzz * Y * Z * Z
      g6 = w_xxx * X * X * X + w_yyy * Y * Y * Y + w_zzz * Z * Z * Z
      g_d = g1 + g2 + g3 + g4 + g5 + g6


      grouped_points = tf.expand_dims(grouped_points, -1)
      g_d = tf.expand_dims(g_d, 3)
      g_d = tf.tile(g_d, [1, 1, 1, in_channels, 1])
      grouped_points = grouped_points * g_d
      grouped_points = tf.reshape(grouped_points, [batch_size, num_point, K_knn, in_channels*taylor_channel])

      feat = conv2d(grouped_points, num_conv, [1,K_knn],
                    padding='VALID', stride=[1,1],
                    bn=bn, is_training=is_training,
                    scope='conv', bn_decay=bn_decay,
                    gn=gn, G=G, is_multi_GPU=is_multi_GPU,
                    activation_fn=activation_fn)

      
      feat = tf.squeeze(feat, axis=[2])

      return feat

def pc_sampling(xyz,
                feat,
                nsample,
                num_point,
                scope='sampling'):
  """ Fully connected layer with non-linear operation.
  
  Args:
    xyz: 3-D tensor B x N x 3
    nsample: k
    num_point: N2
    feat: 3-D tensor B x N x C
  
  Returns:
    feat_sample: 3-D tensor B x N2 x C
  """
  with tf.variable_scope(scope) as sc:
    xyz_new = gather_point(xyz, farthest_point_sample(num_point, xyz))
    _, idx_pooling = knn_point(nsample, xyz, xyz_new)
    
    grouped_points = group_point(feat, idx_pooling)
    feat_sample = tf.nn.max_pool(grouped_points, [1,1,nsample,1], [1,1,1,1], 
    			padding='VALID', data_format='NHWC', name="MAX_POOLING")
    feat_sample = tf.squeeze(feat_sample, axis=[2])

    return feat_sample, xyz_new

def pc_upsampling(xyz_upsample,
                  xyz,
                  feat,
                  scope='upsampling'):
  """ Fully connected layer with non-linear operation.
  
  Args:
    xyz_upsample: 3-D tensor B x N2 x 3
    xyz: 3-D tensor B x N x 3
    feat: 3-D tensor B x N x C
  
  Returns:
    feat_upsample: 3-D tensor B x N2 x C
  """
  with tf.variable_scope(scope) as sc:
    dist, idx_de = three_nn(xyz_upsample, xyz)
    dist = tf.maximum(dist, 1e-10)
    norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
    norm = tf.tile(norm,[1,1,3])
    weight = (1.0/dist) / norm
    feat_upsample = three_interpolate(feat, idx_de, weight)

    return feat_upsample


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None,
                    is_multi_GPU=False,
                    gn=False,
                    G=32):
  """ Fully connected layer with non-linear operation.
  
  Args:
    inputs: 2-D tensor BxN
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
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)
     
    if bn:
      if is_multi_GPU:
        outputs = batch_norm_template_multiGPU(outputs, is_training,
                                               'bn', [0,], bn_decay)
      else:
        outputs = batch_norm_template(outputs, is_training, 
                                      'bn', [0,], bn_decay)
    if gn:
      outputs = group_norm_for_fc(outputs, G=G, scope='gn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
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

def topk_pool(inputs,
              scope,
              k = 2):
  """ top-k pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
  
  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    inputs = tf.transpose(inputs, perm=[0, 2, 1])
    outputs, i_topk = tf.nn.top_k(inputs, k = k, name = sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints
  
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




def group_norm_for_conv(x, G=32, esp=1e-6, scope='gn'):
  with tf.variable_scope(scope) as sc:
    # normalize
    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
    x = tf.transpose(x, [0, 3, 1, 2])
    N, C, H, W = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [N, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = tf.get_variable('gamma', [C],
                            initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable('beta', [C],
                           initializer=tf.constant_initializer(0.0))
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])

    output = tf.reshape(x, [N, C, H, W]) * gamma + beta
    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
    output = tf.transpose(output, [0, 2, 3, 1])

    return output

def group_norm_for_fc(x, G=32, esp=1e-6, scope='gn'):
  with tf.variable_scope(scope) as sc:  
    # normalize
    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
    N, C = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [N, G, C // G])
    mean, var = tf.nn.moments(x, [2], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = tf.get_variable('gamma', [C],
                            initializer=tf.constant_initializer(1.0))
    beta = tf.get_variable('beta', [C],
                           initializer=tf.constant_initializer(0.0))
    gamma = tf.reshape(gamma, [1, C])
    beta = tf.reshape(beta, [1, C])

    output = tf.reshape(x, [N, C]) * gamma + beta
    
    return output



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

def batch_norm_template_multiGPU(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
  
  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.9
  return tf.contrib.layers.batch_norm(inputs, 
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)


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
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
