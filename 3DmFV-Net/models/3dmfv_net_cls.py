import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util


def placeholder_inputs(batch_size, n_points, gmm):

    #Placeholders for the data
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]

    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))

    w_pl = tf.placeholder(tf.float32, shape=(n_gaussians))
    mu_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))
    sigma_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D)) # diagonal
    points_pl = tf.placeholder(tf.float32, shape=(batch_size, n_points, D))

    return points_pl, labels_pl, w_pl, mu_pl, sigma_pl


def get_model(points, w, mu, sigma, is_training, bn_decay=None, weigth_decay=0.005, add_noise=False, num_classes=40):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = points.get_shape()[0].value
    n_points = points.get_shape()[1].value
    n_gaussians = w.shape[0].value
    res = int(np.round(np.power(n_gaussians,1.0/3.0)))


    fv = tf_util.get_3dmfv(points, w, mu, sigma, flatten=False)

    if add_noise:
        noise = tf.cond(is_training,
                        lambda: tf.random_normal(shape=tf.shape(fv), mean=0.0, stddev=0.01, dtype=tf.float32),
                        lambda:  tf.zeros(shape=tf.shape(fv)))
        #noise = tf.random_normal(shape=tf.shape(fv), mean=0.0, stddev=0.01, dtype=tf.float32)
        fv = fv + noise


    grid_fisher = tf.reshape(fv,[batch_size,-1,res,res,res])
    grid_fisher = tf.transpose(grid_fisher, [0, 2, 3, 4, 1])


    # Inception
    layer = 1
    net = inception_module(grid_fisher, n_filters=64,kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = inception_module(net, n_filters=128,kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = inception_module(net, n_filters=256,kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool'+str(layer), stride=[2, 2, 2], padding='SAME')
    layer = layer + 1
    net = inception_module(net, n_filters=256,kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = inception_module(net, n_filters=512,kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    layer = layer + 1
    net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool'+str(layer), stride=[2, 2, 2], padding='SAME')

    net = tf.reshape(net,[batch_size, -1])

    #Classifier
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay, weigth_decay=weigth_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay, weigth_decay=weigth_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay, weigth_decay=weigth_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp3')
    net = tf_util.fully_connected(net, num_classes, activation_fn=None, scope='fc4', is_training=is_training, weigth_decay=weigth_decay)

    return net, fv

def inception_module(input, n_filters=64, kernel_sizes=[3,5], is_training=None, bn_decay=None, scope='inception'):
    one_by_one =  tf_util.conv3d(input, n_filters, [1,1,1], scope= scope + '_conv1',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    three_by_three = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope= scope + '_conv2',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    five_by_five = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[1], kernel_sizes[1], kernel_sizes[1]], scope=scope + '_conv3',
                          stride=[1, 1, 1], padding='SAME', bn=True,
                          bn_decay=bn_decay, is_training=is_training)
    average_pooling = tf_util.avg_pool3d(input, [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope=scope+'_avg_pool', stride=[1, 1, 1], padding='SAME')
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1,1,1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([ one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    return output



def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)

    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    return classify_loss

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print (outputs)
