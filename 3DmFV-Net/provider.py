import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import tensorflow as tf
from sklearn.neighbors import KDTree
import random

# Download dataset for point cloud classification
# DATA_DIR = os.path.join(BASE_DIR, 'data')
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#     www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#     zipfile = os.path.basename(www)
#     os.system('wget %s; unzip %s' % (www, zipfile))
#     os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#     os.system('rm %s' % (zipfile))

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def translate_point_cloud(batch_data, tval = 0.2):
    """ Randomly translate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, translated batch of point clouds
    """
    n_batches = batch_data.shape[0]
    n_points = batch_data.shape[1]
    translation = np.random.uniform(-tval, tval, size=[n_batches,3])
    translation = np.tile(np.expand_dims(translation,1),[1,n_points,1])
    batch_data = batch_data + translation
    # for k in xrange(n_batches):
    #     batch_data[k, ...] = batch_data[k, ...] + translation[k]
    return batch_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 128 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_x_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along x direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 128 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, cosval, -sinval],
                                    [0, sinval, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def scale_point_cloud(batch_data, smin = 0.66, smax = 1.5):
    """ Randomly scale the point clouds to augument the dataset
        scale is per shape
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, scaled batch of point clouds
    """
    scaled = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        sx = np.random.uniform(smin, smax)
        sy = np.random.uniform(smin, smax)
        sz = np.random.uniform(smin, smax)
        scale_matrix = np.array([[sx, 0, 0],
                                    [0, sy, 0],
                                    [0, 0, sz]])
        shape_pc = batch_data[k, ...]
        scaled[k, ...] = np.dot(shape_pc.reshape((-1, 3)), scale_matrix)
    return scaled


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def insert_outliers_to_point_cloud(batch_data, outlier_ratio=0.05):
    """ inserts log_noise Randomly distributed in the unit sphere
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array,  batch of point clouds with log_noise
    """
    B, N, C = batch_data.shape
    outliers = np.random.uniform(-1, 1, [B, int(np.floor(outlier_ratio * N)), C])
    points_idx = np.random.choice(range(0, N), int(np.ceil(N * (1 - outlier_ratio))))
    outlier_data = np.concatenate([batch_data[:, points_idx, :], outliers], axis=1)
    return outlier_data


def occlude_point_cloud(batch_data, occlusion_ratio):
    """ Randomly k remove points (number of points defined by the ratio.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          Bx(N-k)x3 array, occluded batch of point clouds
    """
    B, N, C = batch_data.shape
    k = int(np.round(N*occlusion_ratio))
    occluded_batch_point_cloud = []
    for i in range(B):
        point_cloud = batch_data[i, :, :]
        kdt = KDTree(point_cloud, leaf_size=30, metric='euclidean')
        center_of_occlusion = random.choice(point_cloud)
        #occluded_points_idx = kdt.query_radius(center_of_occlusion.reshape(1, -1), r=occlusion_radius)
        _, occluded_points_idx = kdt.query(center_of_occlusion.reshape(1, -1), k=k)
        point_cloud = np.delete(point_cloud, occluded_points_idx, axis=0)
        occluded_batch_point_cloud.append(point_cloud)
    return np.array(occluded_batch_point_cloud)



def starve_gaussians(batch_data, gmm, starv_coef=0.6, n_points=1024):
    """ sample points from a point cloud with specific sparse regions (defined by the gmm gaussians)
        Input:
          batch_data: BxNx3 array, original batch of point clouds
          gmm: gausian mixture model
        Return:
          BxNx3 array, jittered batch of point clouds
    """

    B, N, D = batch_data.shape
    n_gaussians = len(gmm.weights_)
    choices = [1, starv_coef]
    mu = gmm.means_
    #find a gaussian for each point
    mu = np.tile(np.expand_dims(np.expand_dims(mu,0),0),[B,N,1,1]) #B X N X n_gaussians X D
    batch_data_per_gaussian = np.tile(np.expand_dims(batch_data,-2),[1, 1, n_gaussians, 1] )
    d = np.sum(np.power(batch_data_per_gaussian-mu,2), -1)
    idx = np.argmin(d, axis=2)

    #compute servival probability
    rx = np.random.rand(B, N)
    sk = np.random.choice(choices, n_gaussians)
    p = sk[idx] * rx
    starved_points = []
    for i in range(B):
        topmostidx = np.argsort(p[i,:])[::-1][:n_points]
        starved_points.append(batch_data[i,topmostidx,:])
    return np.asarray(starved_points)


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename, compensate=False, unify=False):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]

    if compensate == True:
        # compensate for problematic cases
        increase_classes = [33, 24, 15] #table, night_stand, plant
        percentage = 3
        idxs = np.squeeze(np.where(np.squeeze(label) == increase_classes))
        n_models = len(idxs)
        if n_models >0:
            n_models_to_add = np.maximum(int(np.round(n_models * (1 + percentage))) - n_models, 1)
            idxs = np.random.choice(idxs, n_models_to_add)
            data = np.concatenate([data, data[idxs,:]])
            label = np.concatenate([label, label[idxs]])
    if unify:
        problem_classes = [33, 23, 15]  # table, night_stand, plant
        alternative_classes = [12, 14, 26] #desk, dresser, flower_pot
        label = replace_labels(np.squeeze(label), problem_classes, alternative_classes)
        label = np.expand_dims(label,-1)
    return (data, label)

def replace_labels(numbers, problem_numbers, alternative_numbers):
    # Replace values
    problem_numbers = np.asarray(problem_numbers)
    alternative_numbers = np.asarray(alternative_numbers)
    n_min, n_max = numbers.min(), numbers.max()
    replacer = np.arange(n_min, n_max + 1)
    mask = problem_numbers <= n_max  # Discard replacements out of range
    replacer[problem_numbers[mask] - n_min] = alternative_numbers[mask]
    numbers = replacer[numbers - n_min]
    return numbers


def loadDataFile(filename, compensate=False, unify=False):
    return load_h5(filename, compensate, unify)

def load_single_model(model_idx = 0,test_train = 'train', file_idxs=0, num_points = 1024):

    if test_train == 'train':
        FILES = getDataFiles( \
            os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    else:
        FILES = getDataFiles( \
            os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    all_models_points, all_models_labels = loadDataFile(FILES[file_idxs])
    points = all_models_points[model_idx, 0:num_points,:]
    labels = all_models_labels[model_idx]
    return np.squeeze(points), labels

def load_single_model_class(clas = 'table',ind=0,test_train = 'train', file_idxs=0, num_points = 1024, n_classes=40):

    shape_names = getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet' + str(n_classes) + '_ply_hdf5_2048/shape_names.txt'))
    shape_dict = {shape_names[i]: i for i in range(len(shape_names))}

    if test_train == 'train':
        FILES = getDataFiles( \
            os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
    else:
        FILES = getDataFiles( \
            os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    all_models_points, all_models_labels = loadDataFile(FILES[file_idxs])
    if  isinstance(clas,basestring):
        idxs = np.squeeze(np.where(np.squeeze(all_models_labels) == shape_dict[clas]))
    else:
        idxs = np.squeeze(np.where(np.squeeze(all_models_labels) == clas))

    if not idxs.size:
        raise ValueError("No such class in this file")
    else:
        idx = idxs[ind]

    points = all_models_points[idx, 0:num_points,:]
    return np.squeeze(points)

def load_dataset(num_points = 1024):

    files = ['train', 'test']

    for test_train in files:
        FILES = getDataFiles( \
            os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/' + test_train + '_files.txt'))

        for fn in range(len(FILES)):
            all_models_points, labels = loadDataFile(FILES[fn])

            if test_train == 'train':
                train_points = all_models_points[:, 0:num_points,:] if fn==0 else np.concatenate([train_points, all_models_points[:, 0:num_points,:]])
                train_labels = labels if fn == 0 else np.concatenate([train_labels, labels])
            else:
                test_points = all_models_points[:, 0:num_points, :] if fn == 0 else np.concatenate(
                    [test_points, all_models_points[:, 0:num_points, :]])
                test_labels = labels if fn == 0 else np.concatenate([test_labels, labels])

    return train_points, np.squeeze(train_labels), test_points, np.squeeze(test_labels)

def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)
