import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

NUM_CLASSES = 15
BACKGROUND_CLASS = -1

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    mask_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))    
    return pointclouds_pl, labels_pl, mask_pl


def get_model(point_cloud, is_training, bn_decay=None, num_class=NUM_CLASSES):
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

    ###########CLASSIFICATION BRANCH
    # print(l3_xyz.shape)
    # print(l3_points.shape)
    net = tf.reshape(l3_points, [batch_size, -1])
    # print(net.shape)
    # print()
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # print("Classification feature vector")
    class_vector = tf.expand_dims(net, axis=1)
    # print(class_vector.shape)
    # print()
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    class_pred = tf_util.fully_connected(net, num_class, activation_fn=None, scope='fc3')

    ###########SEGMENTATION BRANCH
    # Feature Propagation layers
    l3_points_concat = tf.concat([l3_points, class_vector], axis=2)

    # l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points_concat, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, class_vector, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer3')

    # FC layers
    # print(l0_points.shape)
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='seg_fc1', bn_decay=bn_decay)
    # print(net.shape)
    # print()
    end_points['feats'] = net 
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='seg_dp1')
    seg_pred = tf_util.conv1d(net, 2, 1, padding='VALID', activation_fn=None, scope='seg_fc2')
    # print(seg_pred.shape)
    # exit()

    # print(class_pred.shape)
    # print(seg_pred.shape)
    # exit()

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

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
