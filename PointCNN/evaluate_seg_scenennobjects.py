'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
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
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider
sys.path.append(os.path.join(BASE_DIR, '..'))
import data_utils
import pc_util
import pointfly as pf
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointcnn_seg', help='Model name. [default: pointnet2_cls_ssg]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')

parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump/', help='dump folder path [dump]')
parser.add_argument('--with_bg', default = True, help='Whether to have background or not [default: True]')
parser.add_argument('--norm', default = True, help='Whether to normalize data or not [default: False]')
parser.add_argument('--center_data', default = True, help='Whether to explicitly center the data [default: False]')
parser.add_argument('--seg_weight', type=int, default=0.5, help='Segmentation weight in loss')

parser.add_argument('--test_file', default = 'h5_files/main_split/test_objectdataset_augmentedrot_scale75.h5', help='Location of test file')

parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--visu', default = False, help='Whether to dump image for error case [default: False]')
parser.add_argument('--visu_mask', default = False, help='Whether to dump mask [default: False]')
parser.add_argument('--setting', '-x', default = 'object_dataset_x3', help='Setting to use')
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

setting_path = os.path.join(os.path.dirname(__file__), FLAGS.model)
sys.path.append(setting_path)
setting = importlib.import_module(FLAGS.setting)
rotation_range = setting.rotation_range
rotation_range_val = setting.rotation_range_val
scaling_range = setting.scaling_range
scaling_range_val = setting.scaling_range_val
jitter_val = setting.jitter_val

NUM_CLASSES = 15
SHAPE_NAMES = [line.rstrip() for line in \
    open( '../training_data/shape_names_ext.txt')] 

HOSTNAME = socket.gethostname()

np.random.seed(0)

print("Normalized: "+str(NORMALIZED))
print("Center Data: "+str(CENTER_DATA))

TEST_DATA, TEST_LABELS, TEST_MASKS = data_utils.load_withmask_h5(TEST_FILE)      
TEST_MASKS = data_utils.convert_to_binary_mask(TEST_MASKS) 

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
        xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
        rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
        jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
        global_step = tf.Variable(0, trainable=False, name='global_step')
        is_training_pl = tf.placeholder(tf.bool, name='is_training')

        pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3), name='data')
        labels_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE), name='label')
        masks_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINT), name='mask')

        points_augmented = pf.augment(pointclouds_pl, xforms, jitter_range)
        net = MODEL.Net(points=points_augmented, features=None, is_training=is_training_pl, setting=setting)
        classification_logits = net.classification_logits
        segmentation_logits = net.segmentation_logits
        probs = tf.nn.softmax(classification_logits, name='probs')

        ##classification loss
        labels_2d = tf.expand_dims(labels_pl, axis=-1, name='labels_2d')
        labels_tile = tf.tile(labels_2d, (1, tf.shape(classification_logits)[1]), name='labels_tile')
        classify_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels_tile, logits=classification_logits))

        ##segmentation loss
        per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=segmentation_logits, labels=masks_pl), axis=1)
        seg_loss = tf.reduce_mean(per_instance_seg_loss)

        loss_op = (1-SEG_WEIGHT)*classify_loss + SEG_WEIGHT*seg_loss    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'masks_pl': masks_pl,
           'is_training_pl': is_training_pl,
           'pred': probs,
           'seg_pred': segmentation_logits,
           'classify_loss': classify_loss,
           'seg_loss': seg_loss,           
           'loss': loss_op,
           'xforms': xforms,
           'rotations': rotations,
           'jitter_range': jitter_range}

    eval_one_epoch(sess, ops, num_votes)

def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_correct_seg = 0    
    classify_loss_sum = 0
    seg_loss_sum = 0       
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')

    current_data, current_label, current_mask = data_utils.get_current_data_withmask_h5(TEST_DATA, TEST_LABELS, TEST_MASKS, NUM_POINT)

    current_label = np.squeeze(current_label)
    current_mask = np.squeeze(current_mask)

    num_batches = current_data.shape[0]//BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        cur_batch_size = end_idx - start_idx
        
        # Aggregating BEG
        batch_loss_sum = 0 # sum of losses for the batch
        batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
        batch_seg_sum = np.zeros((cur_batch_size, NUM_POINT, 2)) # score for classes
        batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
        for vote_idx in range(num_votes):
            rotated_data = provider.rotate_point_cloud_by_angle(current_data[start_idx:end_idx, :, :],
                                              vote_idx/float(num_votes) * np.pi * 2)

            xforms_np, rotations_np = pf.get_xforms(BATCH_SIZE,
                                                    rotation_range=rotation_range_val,
                                                    scaling_range=scaling_range_val,
                                                    order=setting.rotation_order)

            # Augment batched point clouds by rotation and jittering
            feed_dict = {ops['pointclouds_pl']: rotated_data,
                        ops['labels_pl']: current_label[start_idx:end_idx],
                        ops['masks_pl']: current_mask[start_idx:end_idx],
                        ops['is_training_pl']: is_training,
                        ops['xforms']: xforms_np,
                        ops['rotations']: rotations_np,
                        ops['jitter_range']: np.array([jitter_val])}

            loss_val, pred_val, seg_val, classify_loss, seg_loss = sess.run([ops['loss'], ops['pred'], ops['seg_pred'], ops['classify_loss'], ops['seg_loss']],
                                      feed_dict=feed_dict)

            pred_val = np.sum(pred_val, axis=1)
            # pred_val = np.argmax(pred_val, 1)

            batch_pred_sum += pred_val
            batch_seg_sum += seg_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
        # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
        # pred_val = np.argmax(batch_pred_classes, 1)
        pred_val = np.argmax(batch_pred_sum, 1)
        # Aggregating END
        
        correct = np.sum(pred_val == current_label[start_idx:end_idx])

        seg_val = np.argmax(batch_seg_sum, 2)
        seg_correct = np.sum(seg_val == current_mask[start_idx:end_idx])
        total_correct_seg += seg_correct

        # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
        total_correct += correct
        total_seen += cur_batch_size
        loss_sum += batch_loss_sum

        for i in range(start_idx, end_idx):
            l = current_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i-start_idx] == l)
            fout.write('%s, %s\n' % (SHAPE_NAMES[pred_val[i-start_idx]], SHAPE_NAMES[l]))

            gt_mask = current_mask[i]
            pred_mask = seg_val[i-start_idx]

            pred_mask_idx = np.where(pred_mask==1)[0]
            gt_mask_idx = np.where(gt_mask==1)[0]
            correct_obj_mask = np.where((pred_mask==gt_mask) & (pred_mask==1))[0]

            if (len(correct_obj_mask)==1):
                continue

            if (i%20==0 and FLAGS.visu_mask):
                ###1)
                img_filename = '%d_label_%s_pred_%s_gtmask.jpg' % (i, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                img_filename = os.path.join(DUMP_DIR, img_filename)
                output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, gt_mask_idx, :]))
                scipy.misc.imsave(img_filename, output_img)

                #save ply
                ply_filename = '%d_label_%s_pred_%s_gtmask.ply' % (i, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                ply_filename = os.path.join(DUMP_DIR, ply_filename)
                data_utils.save_ply(np.squeeze(current_data[i, gt_mask_idx, :]),ply_filename)                   

                ###2)
                img_filename = '%d_label_%s_pred_%s_predmask.jpg' % (i, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                img_filename = os.path.join(DUMP_DIR, img_filename)
                output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, pred_mask_idx, :]))
                scipy.misc.imsave(img_filename, output_img)

                #save ply
                ply_filename = '%d_label_%s_pred_%s_predmask.ply' % (i, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                ply_filename = os.path.join(DUMP_DIR, ply_filename)
                data_utils.save_ply(np.squeeze(current_data[i, pred_mask_idx, :]),ply_filename) 

                ###3)
                img_filename = '%d_label_%s_pred_%s_correctpredmask.jpg' % (i, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                img_filename = os.path.join(DUMP_DIR, img_filename)
                output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, correct_obj_mask, :]))
                scipy.misc.imsave(img_filename, output_img)

                #save ply
                ply_filename = '%d_label_%s_pred_%s_correctpredmask.ply' % (i, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                ply_filename = os.path.join(DUMP_DIR, ply_filename)
                data_utils.save_ply(np.squeeze(current_data[i, correct_obj_mask, :]),ply_filename)                 

            
            if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                img_filename = os.path.join(DUMP_DIR, img_filename)
                output_img = pc_util.point_cloud_three_views(np.squeeze(current_data[i, :, :]))
                scipy.misc.imsave(img_filename, output_img)

                #save ply
                ply_filename = '%d_label_%s_pred_%s.ply' % (error_cnt, SHAPE_NAMES[l],
                                                       SHAPE_NAMES[pred_val[i-start_idx]])
                data_utils.save_ply(np.squeeze(current_data[i, :, :]),ply_filename)                
                error_cnt += 1

    log_string('total seen: %d' % (total_seen)) 
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    log_string('seg accuracy: %f' % (total_correct_seg / (float(total_seen)*NUM_POINT)))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
