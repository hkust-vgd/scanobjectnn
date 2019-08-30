import os
import sys
import numpy as np

import importlib
import argparse
import tensorflow as tf
import socket
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import provider
import utils
sys.path.append(os.path.join(BASE_DIR, '..'))
import data_utils
from mapping2 import *

NUM_CLASSES = 15

augment_rotation, augment_scale, augment_translation, augment_jitter, augment_outlier = (False, True, True, True, False)

parser = argparse.ArgumentParser()
#Parameters for learning
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='3dmfv_net_cls', help='Model name [default: 3dmfv_net_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')

parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump_synthetic_trained_on_real/', help='dump folder path [dump]')
parser.add_argument('--with_bg', default = True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default = True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--center_data', default = True, help='Whether to explicitly center the data [default: False]')
parser.add_argument('--num_class', type=int, default = 15, help='Number of classes to classify.')

parser.add_argument('--test_file', default = 'modelnet/modelnet_test.h5', help='Location of test file')

# Parameters for GMM
parser.add_argument('--gmm_type',  default='grid', help='type of gmm [grid/learn], learn uses expectation maximization algorithm (EM) [default: grid]')
parser.add_argument('--num_gaussians', type=int , default=5, help='number of gaussians for gmm, if grid specify subdivisions, if learned specify actual number[default: 5, for grid it means 125 gaussians]')
parser.add_argument('--gmm_variance', type=float,  default=0.04, help='variance for grid gmm, relevant only for grid type')
FLAGS = parser.parse_args()


N_GAUSSIANS = FLAGS.num_gaussians
GMM_TYPE = FLAGS.gmm_type
GMM_VARIANCE = FLAGS.gmm_variance

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


def evaluate(gmm, num_votes=1):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            points_pl, labels_pl, w_pl, mu_pl, sigma_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, gmm )
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, fv = MODEL.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl, num_classes=NUM_CLASSES)
            loss = MODEL.get_loss(pred, labels_pl)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': points_pl,
               'labels_pl': labels_pl,
               'w_pl': w_pl,
               'mu_pl': mu_pl,
               'sigma_pl': sigma_pl,
               'is_training_pl': is_training_pl,
               'fv': fv,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops, gmm, num_votes)


def eval_one_epoch(sess, ops, gmm, num_votes):
    """ ops: dict mapping from string to tf ops """
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
        end_idx = (batch_idx + 1) * BATCH_SIZE
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
                         ops['w_pl']: gmm.weights_,
                         ops['mu_pl']: gmm.means_,
                         ops['sigma_pl']: np.sqrt(gmm.covariances_),                         
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)

            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
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


if __name__ == "__main__":

    gmm = utils.get_3d_grid_gmm(subdivisions=[N_GAUSSIANS, N_GAUSSIANS, N_GAUSSIANS], variance=GMM_VARIANCE)
    LOG_DIR = MODEL_PATH[:MODEL_PATH.rfind('/')]
    gmm = pickle.load(open(os.path.join(LOG_DIR,'gmm.p'), "rb"))
    evaluate(gmm, num_votes=1)
    #export_visualizations(gmm, LOG_DIR,n_model_limit=None)

    LOG_FOUT.close()


