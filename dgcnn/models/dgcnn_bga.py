import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net

# NUM_CLASSES = 20
# NUM_CLASSES = 10
# NUM_CLASSES = 15
# NUM_CLASSES = 40

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))    
    mask_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, mask_pl


def get_model(point_cloud, is_training, bn_decay=None, num_class=NUM_CLASSES):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  adj_matrix = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

  point_cloud_transformed = tf.matmul(point_cloud, transform)
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net
 
  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
  
  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg', bn_decay=bn_decay)
 
  #For segmentation branch
  out_max = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
  print("Out Max")
  print(out_max.shape)
  expand = tf.tile(out_max, [1, num_point, 1, 1])
  print("Out Max expanded")
  print(expand.shape)

  net = tf.reduce_max(net, axis=1, keep_dims=True)

  ## CLASSIFICATION BRANCH
  net = tf.reshape(net, [batch_size, -1]) 
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)

  class_vector = tf.expand_dims(net, axis = 1)
  class_vector = tf.expand_dims(class_vector, axis = 1)

  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  class_pred = tf_util.fully_connected(net, num_class, activation_fn=None, scope='fc3')

  ## SEGMENTATION BRANCH
  print("Class Vector")
  print(class_vector.shape)
  # exit()
  
  class_vector_expanded = tf.tile(class_vector, [1, num_point, 1, 1])

  concat = tf.concat([class_vector_expanded, expand, net1, net2, net3, net4], axis=-1)
  net = tf_util.conv2d(concat, 512, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv1', is_dist=True)
  net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
             bn=True, is_training=is_training, scope='seg/conv2', is_dist=True)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
  net = tf_util.conv2d(net, 2, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv3', is_dist=True)
  seg_pred = tf.squeeze(net, [2])  

  return class_pred, seg_pred


def get_loss(class_pred, seg_pred, gt_label, gt_mask, seg_weight = 0.5):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = gt_mask.shape[0]
    num_point = gt_mask.shape[1]

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_pred, labels=gt_label)
    classify_loss = tf.reduce_mean(loss)

    #mask loss
    ###convert mask to binary mask
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=gt_mask), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    total_loss = (1-seg_weight)*classify_loss + seg_weight*seg_loss

    return total_loss, classify_loss, seg_loss











