import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
sys.path.append(os.path.join(BASE_DIR, '..'))
import data_utils
from mapping2 import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='spidercnn_cls_xyz', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--num_class', type=int, default = 15, help='Number of classes to classify.')

parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump_synthetic_trained_on_real/', help='dump folder path [dump]')
parser.add_argument('--with_bg', default = True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default = True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--center_data', default = False, help='Whether to explicitly center the data [default: False]')

parser.add_argument('--test_file', default = 'modelnet/modelnet_test.h5', help='Location of test file')

FLAGS = parser.parse_args()


DATA_DIR = os.path.join(ROOT_DIR, '../../../../')
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

WITH_BG = FLAGS.with_bg
NORMALIZED = FLAGS.norm
TEST_FILE = FLAGS.test_file
CENTER_DATA = FLAGS.center_data

NUM_C = 15
SHAPE_NAMES = [line.rstrip() for line in \
    open( '../training_data/shape_names_ext.txt')] 

HOSTNAME = socket.gethostname()


np.random.seed(0)

NUM_CLASSES = FLAGS.num_class
print("Number of Classes: "+str(NUM_CLASSES))
print("Normalized: "+str(NORMALIZED))
print("Center Data: "+str(CENTER_DATA))

if (".h5" in TEST_FILE):
    TEST_DATA, TEST_LABELS = data_utils.load_h5(TEST_FILE)
else:
    TEST_DATA, TEST_LABELS = data_utils.load_data(TEST_FILE, NUM_POINT, with_bg_pl = WITH_BG)    

if (CENTER_DATA):
    TEST_DATA = data_utils.center_data(TEST_DATA)

if (NORMALIZED):
    TEST_DATA = data_utils.normalize_data(TEST_DATA)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = MODEL.get_model(pointclouds_pl, is_training_pl, num_class=NUM_CLASSES)
        loss = MODEL.get_loss(pred, labels_pl)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_C)]
    total_correct_class = [0 for _ in range(NUM_C)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')

    if (".h5" in TEST_FILE):
        current_data, current_label = data_utils.get_current_data_h5(TEST_DATA, TEST_LABELS, NUM_POINT)
    else:
        current_data, current_label = data_utils.get_current_data(TEST_DATA, TEST_LABELS, NUM_POINT)

    current_label = np.squeeze(current_label)

    ####################################################
    print(current_data.shape)
    print(current_label.shape)

    filtered_data = []
    filtered_label = []
    for i in range(current_label.shape[0]):
        if (current_label[i] in MODELNET_TO_OBJECTDATASET.keys()):
            filtered_label.append(current_label[i])
            filtered_data.append(current_data[i,:])

    filtered_data = np.array(filtered_data)
    filtered_label = np.array(filtered_label)
    print(filtered_data.shape)
    print(filtered_label.shape)

    current_data = filtered_data
    current_label = filtered_label
    ###################################################    

    num_batches = current_data.shape[0]//BATCH_SIZE

    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        # Aggregating BEG
        batch_loss_sum = 0 # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                              vote_idx/float(num_votes) * np.pi * 2)
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)

            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
        # pred_val = np.argmax(batch_pred_classes, 1)
        pred_val = np.argmax(batch_pred_sum, 1)
        # Aggregating END
        
        for i in range(start_idx, end_idx):
            total_seen += 1
            if (pred_val[i-start_idx] not in OBJECTDATASET_TO_MODELNET.keys()):
                continue
            else:
                possible_label = OBJECTDATASET_TO_MODELNET[pred_val[i-start_idx]]
                if (current_label[i] in possible_label):
                    total_correct +=1

        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[MODELNET_TO_OBJECTDATASET[l]] += 1

            if (pred_val[i-start_idx] in OBJECTDATASET_TO_MODELNET.keys()):
                possible_label = OBJECTDATASET_TO_MODELNET[pred_val[i-start_idx]]
                if (l in possible_label):
                    total_correct_class[MODELNET_TO_OBJECTDATASET[l]] += 1                  


            pred_label = SHAPE_NAMES[pred_val[i-start_idx]]
            # groundtruth_label = SHAPE_NAMES[MODELNET_TO_OBJECTDATASET[l]]
            groundtruth_label = SHAPE_NAMES[MODELNET_TO_OBJECTDATASET[l]]

            fout.write('%s, %s\n' % (pred_label, groundtruth_label))
            
    
    log_string('total seen: %d' % (total_seen)) 
    # log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    seen_class_accuracies = []
    seen_correct_class = []
    for i in range(len(total_seen_class)):
        if total_seen_class[i] != 0 :
            seen_class_accuracies.append(total_seen_class[i])
            seen_correct_class.append(total_correct_class[i])
    log_string('eval avg class acc: %f' % (np.mean(np.array(seen_correct_class)/np.array(seen_class_accuracies,dtype=np.float))))
    
    for i, name in enumerate(SHAPE_NAMES):
        if (total_seen_class[i] == 0):
            accuracy = -1
        else:
            accuracy = total_correct_class[i]/float(total_seen_class[i])
        log_string('%10s:\t%0.3f' % (name, accuracy))

    

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
