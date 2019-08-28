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

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_partseg', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')

parser.add_argument('--model_path', default='../../../../pointnet/log_partseg_augmented25_norot/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump_partseg_augmented25_norot/', help='dump folder path [dump]')
parser.add_argument('--with_bg', default = True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default = True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--center_data', default = True, help='Whether to explicitly center the data [default: False]')
parser.add_argument('--seg_weight', type=int, default=1.0, help='Segmentation weight in loss')

# parser.add_argument('--test_file', default = '../training_data/test_objectdataset_v1.pickle', help='Location of test file')
parser.add_argument('--test_file', default = '/home/vgd/object_dataset/parts/test_objectdataset_augmented25_norot.h5', help='Location of test file')

parser.add_argument('--visu_mask', default = False, help='Whether to dump mask [default: False]')
parser.add_argument('--visu', default = False, help='Whether to dump image for error case [default: False]')
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
SEG_WEIGHT = FLAGS.seg_weight

NUM_CLASSES = 6
SHAPE_NAMES = [line.rstrip() for line in \
    open( '../training_data/chair_parts.txt')] 

HOSTNAME = socket.gethostname()

np.random.seed(0)

print("Normalized: "+str(NORMALIZED))
print("Center Data: "+str(CENTER_DATA))   

TEST_DATA, TEST_LABELS, TEST_PARTS = data_utils.load_parts_h5(TEST_FILE) 

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
        pointclouds_pl, labels_pl, parts_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        seg_pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        total_loss = MODEL.get_loss(seg_pred, parts_pl, end_points)

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
           'parts_pl': parts_pl,
           'is_training_pl': is_training_pl,
           'seg_pred': seg_pred,
           'loss': total_loss}

    eval_one_epoch(sess, ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False

    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    total_correct_seg = 0

    current_data, current_label, current_parts = data_utils.get_current_data_parts_h5(TEST_DATA, TEST_LABELS, TEST_PARTS, NUM_POINT)

    current_label = np.squeeze(current_label)
    current_parts = np.squeeze(current_parts)

    num_batches = current_data.shape[0]//BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        # Aggregating BEG
        batch_loss_sum = 0 # sum of losses for the batch
        batch_seg_sum = np.zeros((cur_batch_size, NUM_POINT, NUM_CLASSES)) # score for classes
        for vote_idx in range(num_votes):
            # rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
            #                                   vote_idx/float(num_votes) * np.pi * 2)
            rotated_data = current_data[start_idx:end_idx, :, :]           
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                        ops['parts_pl']: current_parts[start_idx:end_idx],
                        ops['labels_pl']: current_label[start_idx:end_idx],
                        ops['is_training_pl']: is_training}
            loss_val, seg_val = sess.run([ops['loss'], ops['seg_pred']],
                                      feed_dict=feed_dict)

            batch_seg_sum += seg_val
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
        # pred_val = np.argmax(batch_pred_classes, 1)
        # Aggregating END
        seg_val = np.argmax(batch_seg_sum, 2)
        seg_correct = np.sum(seg_val == current_parts[start_idx:end_idx])
        total_correct_seg += seg_correct
        
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum

        for i in range(start_idx, end_idx):
            parts = current_parts[i]
            for j in range(len(parts)):
                part = parts[j]

                total_seen_class[part] += 1
                total_correct_class[part] += (seg_val[i-start_idx][j] == part)

        total_parts_seen = 0
        cum_sum = 0
        part_accs = []
        for i in range(NUM_CLASSES):
            if (total_seen_class[i]==0):
                part_accs.append(-1.0)
                continue
            part_acc = float(total_correct_class[i])/float(total_seen_class[i])
            cum_sum += part_acc
            part_accs.append(part_acc)
            total_parts_seen +=1

    log_string('total seen: %d' % (total_seen)) 
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval seg accuracy: %f' % (total_correct_seg / (float(total_seen)*NUM_POINT)))
    log_string('eval avg class acc: %f' % (cum_sum/float(total_parts_seen)))  

    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, part_accs[i]))
    


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=1)
    LOG_FOUT.close()
