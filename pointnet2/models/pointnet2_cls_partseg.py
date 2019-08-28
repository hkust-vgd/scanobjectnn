import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

NUM_CLASSES = 6

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    mask_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))    
    return pointclouds_pl, labels_pl, mask_pl


def get_model(point_cloud, is_training, bn_decay=None, num_class = NUM_CLASSES):
    """ Part segmentation PointNet, input is BxNx3 (XYZ) """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = None

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=64, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    ###########SEGMENTATION BRANCH
    # Feature Propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='seg_fc1', bn_decay=bn_decay)
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='seg_dp1')
    seg_pred = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='seg_fc2')

    return seg_pred

def tf_count(t, val):
    elements_equal_to_value = tf.equal(t, val)
    as_ints = tf.cast(elements_equal_to_value, tf.int32)
    count = tf.reduce_sum(as_ints)

    return count

def get_loss(seg_pred, gt_seg):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = gt_seg.shape[0]
    num_point = gt_seg.shape[1]

    # ##try adaptive weights
    # count_0 = tf.cast(tf_count(gt_seg, 0), tf.float32)
    # count_2 = tf.cast(tf_count(gt_seg, 2), tf.float32)
    # count_3 = tf.cast(tf_count(gt_seg, 3), tf.float32)
    # count_4 = tf.cast(tf_count(gt_seg, 4), tf.float32)
    # count_5 = tf.cast(tf_count(gt_seg, 5), tf.float32)
    # total_count = tf.cast(count_0 + count_2 + count_3 + count_4 + count_5, tf.float32)
    # labels_one_hot = tf.one_hot(gt_seg, 6, on_value=1.0, off_value=0.0)
    # class_weights = [total_count/count_0, 1.0, total_count/count_2, total_count/count_3, total_count/count_4, total_count/count_5]

    # weights = tf.reduce_sum(class_weights*labels_one_hot, axis=-1)
    # unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=gt_seg)
    # seg_loss = tf.reduce_mean(weights*unweighted_loss)

    ##try weighted loss
    # labels_one_hot = tf.one_hot(gt_seg, 6, on_value=1.0, off_value=0.0)
    # # class_weights = [1.0, 1.0, 10.0, 40.0, 30.0, 10.0]
    # class_weights = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    # weights = tf.reduce_sum(class_weights*labels_one_hot, axis=-1)
    # unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=gt_seg)
    # seg_loss = tf.reduce_mean(weights*unweighted_loss)

    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=gt_seg), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    total_loss = seg_loss

    return total_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
