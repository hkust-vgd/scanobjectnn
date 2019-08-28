import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
sys.path.append(os.path.join(BASE_DIR, '..'))
import data_utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_seg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')

parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--with_bg', default = True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default = True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--center_data', default = True, help='Whether to explicitly center the data [default: False]')
parser.add_argument('--seg_weight', type=int, default=0.5, help='Segmentation weight in loss')

parser.add_argument('--train_file', default = 'h5_files/main_split/training_objectdataset_augmentedrot_scale75.h5', help='Location of training file')
parser.add_argument('--test_file', default = 'h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5', help='Location of test file')

parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

DATA_DIR = os.path.join(ROOT_DIR, '../../../../')
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

WITH_BG = FLAGS.with_bg
NORMALIZED = FLAGS.norm
TRAIN_FILE = FLAGS.train_file
TEST_FILE = FLAGS.test_file
CENTER_DATA = FLAGS.center_data
SEG_WEIGHT = FLAGS.seg_weight

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
os.system('cp ../data_utils.py %s' % (LOG_DIR)) # bkp of data utils
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# MAX_NUM_POINT = 2048
# NUM_CLASSES = 40
# NUM_CLASSES = 20
# NUM_CLASSES = 10
NUM_CLASSES = 15

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

print("Normalized: "+str(NORMALIZED))
print("Center Data: "+str(CENTER_DATA))


TRAIN_DATA, TRAIN_LABELS, TRAIN_MASKS = data_utils.load_withmask_h5(TRAIN_FILE)
TEST_DATA, TEST_LABELS, TEST_MASKS = data_utils.load_withmask_h5(TEST_FILE)   

if (CENTER_DATA):
    TRAIN_DATA = data_utils.center_data(TRAIN_DATA)
    TEST_DATA = data_utils.center_data(TEST_DATA)

if (NORMALIZED):
    TRAIN_DATA = data_utils.normalize_data(TRAIN_DATA)
    TEST_DATA = data_utils.normalize_data(TEST_DATA)

TRAIN_MASKS = data_utils.convert_to_binary_mask(TRAIN_MASKS)
TEST_MASKS = data_utils.convert_to_binary_mask(TEST_MASKS)

print(len(TRAIN_DATA))
print(len(TEST_DATA))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            class_pred, seg_pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            total_loss, classify_loss, seg_loss = MODEL.get_loss(class_pred, seg_pred, labels_pl, masks_pl, end_points, seg_weight=SEG_WEIGHT)
            tf.summary.scalar('loss', total_loss)
            tf.summary.scalar('classify_loss', classify_loss)
            tf.summary.scalar('seg_loss', seg_loss)

            correct = tf.equal(tf.argmax(class_pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            seg_correct = tf.equal(tf.argmax(seg_pred, 2), tf.to_int64(masks_pl))
            seg_accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / (float(BATCH_SIZE)*NUM_POINT)
            tf.summary.scalar('seg_accuracy', seg_accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        #Load checkpoint
        # saver.restore(sess, os.path.join(LOG_DIR,'model.ckpt'))
        # log_string("Model restored.")


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': class_pred,
               'seg_pred': seg_pred,
               'loss': total_loss,
               'classify_loss': classify_loss,
               'seg_loss': seg_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    current_data, current_label, current_mask = data_utils.get_current_data_withmask_h5(TRAIN_DATA, TRAIN_LABELS, TRAIN_MASKS, NUM_POINT)

    current_label = np.squeeze(current_label)
    current_mask = np.squeeze(current_mask)

    num_batches = current_data.shape[0]//BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_correct_seg = 0    
    classify_loss_sum = 0
    seg_loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        feed_dict = {ops['pointclouds_pl']: jittered_data,
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['masks_pl']: current_mask[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val, seg_val, classify_loss, seg_loss = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred'], ops['seg_pred'], ops['classify_loss'], ops['seg_loss']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        
        seg_val = np.argmax(seg_val, 2)
        seg_correct = np.sum(seg_val == current_mask[start_idx:end_idx])
        total_correct_seg += seg_correct

        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        classify_loss_sum += classify_loss
        seg_loss_sum += seg_loss
    
    log_string('mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('classify mean loss: %f' % (classify_loss_sum / float(num_batches)))
    log_string('seg mean loss: %f' % (seg_loss_sum / float(num_batches)))
    log_string('accuracy: %f' % (total_correct / float(total_seen)))
    log_string('seg accuracy: %f' % (total_correct_seg / (float(total_seen)*NUM_POINT)))
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    classify_loss_sum = 0
    seg_loss_sum = 0
    total_correct_seg = 0

    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # data_utils.shuffle_points(TEST_DATA)
    # current_data, current_label = data_utils.get_current_data(TEST_DATA, TEST_LABELS, NUM_POINT)
    # current_data, current_label = data_utils.get_current_data_h5(TEST_DATA, TEST_LABELS, NUM_POINT)
    current_data, current_label, current_mask = data_utils.get_current_data_withmask_h5(TEST_DATA, TEST_LABELS, TEST_MASKS, NUM_POINT)

    current_label = np.squeeze(current_label)
    current_mask = np.squeeze(current_mask)

    num_batches = current_data.shape[0]//BATCH_SIZE
        
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['masks_pl']: current_mask[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val, seg_val, classify_loss, seg_loss = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred'], ops['seg_pred'], ops['classify_loss'], ops['seg_loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        seg_val = np.argmax(seg_val, 2)
        seg_correct = np.sum(seg_val == current_mask[start_idx:end_idx])
        total_correct_seg += seg_correct
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += (loss_val*BATCH_SIZE)
        classify_loss_sum += classify_loss*BATCH_SIZE
        seg_loss_sum += seg_loss*BATCH_SIZE        
        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval classify mean loss: %f' % (classify_loss_sum / float(total_seen)))
    log_string('eval seg mean loss: %f' % (seg_loss_sum / float(total_seen)))

    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))    
    log_string('eval seg accuracy: %f' % (total_correct_seg / (float(total_seen)*NUM_POINT)))

if __name__ == "__main__":
    train()
    LOG_FOUT.close()
