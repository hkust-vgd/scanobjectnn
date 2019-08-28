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

import itertools
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='spidercnn_cls_xyz', help='Model name: dgcnn [default: dgcnn]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')

parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='confusion_matrix/', help='dump folder path [dump]')
parser.add_argument('--with_bg', default = True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default = True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--center_data', default = False, help='Whether to explicitly center the data [default: False]')

parser.add_argument('--test_file', default = 'h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5', help='Location of test file')

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

NUM_CLASSES = 15
SHAPE_NAMES = [line.rstrip() for line in \
    open( '../training_data/shape_names_ext.txt')] 

HOSTNAME = socket.gethostname()


np.random.seed(0)

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
        pred = MODEL.get_model(pointclouds_pl, is_training_pl)
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
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')

    if (".h5" in TEST_FILE):
        current_data, current_label = data_utils.get_current_data_h5(TEST_DATA, TEST_LABELS, NUM_POINT)
    else:
        current_data, current_label = data_utils.get_current_data(TEST_DATA, TEST_LABELS, NUM_POINT)

    current_label = np.squeeze(current_label)

    num_batches = current_data.shape[0]//BATCH_SIZE

    current_pred = []
    
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
        
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
        total_correct += correct
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum

        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
            fout.write('%s, %s\n' % (SHAPE_NAMES[pred_val[i-start_idx]], SHAPE_NAMES[l]))
            
            current_pred.append(pred_val[i-start_idx])
    
    log_string('total seen: %d' % (total_seen)) 
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    
    #Plot confusion matrix
    current_pred = np.array(current_pred)
    groundtruth = current_label.flatten()
    predictions = current_pred.flatten()

    mat = confusion_matrix(groundtruth, predictions)

    plt.style.use('seaborn-paper')
    plt.rcParams["figure.figsize"] = (10,10)
    ax = plt.subplot(111)
    cmap = plt.cm.Reds
    mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    mat = np.nan_to_num(mat, copy=True)

    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    # cbar = plt.colorbar(fraction=0.03, pad=0.05, aspect=30)
    # cbar.ax.tick_params(labelsize=10)
    tick_marks = np.arange(len(SHAPE_NAMES))
    plt.xticks(tick_marks, SHAPE_NAMES, rotation=90)
    plt.yticks(tick_marks, SHAPE_NAMES)

    plt.ylabel('Ground truth')
    plt.xlabel('Prediction')

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)

    plt.tight_layout()
    plt.savefig(os.path.join(DUMP_DIR,'matrix.pdf'))
    plt.show()       

if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
