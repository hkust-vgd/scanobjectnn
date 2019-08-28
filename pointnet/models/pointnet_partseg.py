import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

NUM_CLASSES = 6

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))    
    seg_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, seg_pl


def get_model(point_cloud, is_training, bn_decay=None, num_class = NUM_CLASSES):
    """ Classification PointNet, input is BxNx3, output BxNx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    print(point_feat)

    net = tf_util.conv2d(point_feat, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    #for part_segmentation
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat([point_feat, global_feat_expand], 3)
    print(concat_feat)

    net = tf_util.conv2d(concat_feat, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)

    net = tf_util.conv2d(net, num_class, [1,1],
                         padding='VALID', stride=[1,1], activation_fn=None,
                         scope='conv10')    

    seg_pred = tf.squeeze(net, [2]) # BxNxC

    return seg_pred, end_points


def get_loss(seg_pred, gt_seg, end_points, reg_weight=0.001):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = gt_seg.shape[0]
    num_point = gt_seg.shape[1]

    #mask loss
    ###convert mask to binary mask
    ##try weighted loss
    # labels_one_hot = tf.one_hot(gt_seg, 6, on_value=1.0, off_value=0.0)
    # class_weights = [1.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    # weights = tf.reduce_sum(class_weights*labels_one_hot, axis=-1)
    # unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=gt_seg)
    # seg_loss = tf.reduce_mean(weights*unweighted_loss)


    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=gt_seg), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 

    total_loss = seg_loss + mat_diff_loss * reg_weight

    return total_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
