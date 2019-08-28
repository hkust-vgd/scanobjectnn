import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import numpy as np
import pc_util
import scipy.misc
import string
import pickle
import plyfile
import h5py

# DATA_PATH = '/media/mikacuy/6TB_HDD/object_dataset_v1_fixed/'

def save_ply(points, filename, colors=None, normals=None):
    vertex = np.core.records.fromarrays(points.transpose(), names='x, y, z', formats='f4, f4, f4')
    n = len(vertex)
    desc = vertex.dtype.descr

    if normals is not None:
        vertex_normal = np.core.records.fromarrays(normals.transpose(), names='nx, ny, nz', formats='f4, f4, f4')
        assert len(vertex_normal) == n
        desc = desc + vertex_normal.dtype.descr

    if colors is not None:
        vertex_color = np.core.records.fromarrays(colors.transpose() * 255, names='red, green, blue',
                                                  formats='u1, u1, u1')
        assert len(vertex_color) == n
        desc = desc + vertex_color.dtype.descr

    vertex_all = np.empty(n, dtype=desc)

    for prop in vertex.dtype.names:
        vertex_all[prop] = vertex[prop]

    if normals is not None:
        for prop in vertex_normal.dtype.names:
            vertex_all[prop] = vertex_normal[prop]

    if colors is not None:
        for prop in vertex_color.dtype.names:
            vertex_all[prop] = vertex_color[prop]

    ply = plyfile.PlyData([plyfile.PlyElement.describe(vertex_all, 'vertex')], text=False)
    #if not os.path.exists(os.path.dirname(filename)):
    #    os.makedirs(os.path.dirname(filename))
    ply.write(filename)

def load_pc_file(filename, suncg = False, with_bg = True):
	#load bin file
	# pc=np.fromfile(filename, dtype=np.float32)
	pc=np.fromfile(os.path.join(DATA_PATH, filename), dtype=np.float32)

	#first entry is the number of points
	#then x, y, z, nx, ny, nz, r, g, b, label, nyu_label
	if(suncg):
		pc = pc[1:].reshape((-1,3))
	else:
		pc = pc[1:].reshape((-1,11))

	#only use x, y, z for now
	if with_bg:
		pc = np.array(pc[:,0:3])
		return pc

	else:
		##To remove backgorund points
		##filter unwanted class
		filtered_idx = np.intersect1d(np.intersect1d(np.where(pc[:,-1]!=0)[0],np.where(pc[:,-1]!=1)[0]), np.where(pc[:,-1]!=2)[0])
		(values, counts) = np.unique(pc[filtered_idx,-1], return_counts=True)
		max_ind = np.argmax(counts)
		idx = np.where(pc[:,-1]==values[max_ind])[0]
		pc = np.array(pc[idx,0:3])
		return pc

def load_data(filename, num_points=1024, suncg_pl = False, with_bg_pl = True):
	with open(filename, 'rb') as handle:
		data = pickle.load(handle)
		print("Data loaded.")

	pcs = []
	labels = []

	print("With BG: "+str(with_bg_pl))
	for i in range(len(data)):
		entry = data[i]
		filename = entry["filename"].replace('objects_bin/','')
		pc = load_pc_file(filename, suncg = suncg_pl, with_bg = with_bg_pl)
		label = entry['label']

		if (pc.shape[0]<num_points):
			continue

		pcs.append(pc)
		labels.append(label)

	print(len(pcs))
	print(len(labels))

	return pcs, labels

def shuffle_points(pcs):
	for pc in pcs:
		np.random.shuffle(pc)
	return pcs

def get_current_data(pcs, labels, num_points):
	sampled = []
	for pc in pcs:
		if(pc.shape[0]<num_points):
			# TODO repeat points
			print("Points too less.")
			return
		else:
			#faster than shuffle_points
			idx = np.arange(pc.shape[0])
			np.random.shuffle(idx)
			sampled.append(pc[idx[:num_points],:])

	sampled = np.array(sampled)
	labels = np.array(labels)

	#shuffle per epoch
	idx = np.arange(len(labels))
	np.random.shuffle(idx)

	sampled = sampled[idx]
	labels = labels[idx]

	return sampled, labels

def normalize_data(pcs):
	for pc in pcs:
		#get furthest point distance then normalize
		d = max(np.sum(np.abs(pc)**2,axis=-1)**(1./2))
		pc /= d

		# pc[:,0]/=max(abs(pc[:,0]))
		# pc[:,1]/=max(abs(pc[:,1]))
		# pc[:,2]/=max(abs(pc[:,2]))

	return pcs

def normalize_data_multiview(pcs, num_view=5):
	pcs_norm = []
	for i in range(len(pcs)):
		pc = []
		for j in range(num_view):
			pc_view = pcs[i][j, :, :]
			d = max(np.sum(np.abs(pc_view)**2,axis=-1)**(1./2))
			pc.append(pc_view/d)
		pc = np.array(pc)
		pcs_norm.append(pc)
	pcs_norm = np.array(pcs_norm)
	print("Normalized")
	print(pcs_norm.shape)
	return pcs_norm


#USE For SUNCG, to center to origin
def center_data(pcs):
	for pc in pcs:
		centroid = np.mean(pc, axis=0)
		pc[:,0]-=centroid[0]
		pc[:,1]-=centroid[1]
		pc[:,2]-=centroid[2]
	return pcs

##For h5 files
def get_current_data_h5(pcs, labels, num_points):
	#shuffle points to sample
	idx_pts = np.arange(pcs.shape[1])
	np.random.shuffle(idx_pts)

	sampled = pcs[:,idx_pts[:num_points],:]
	#sampled = pcs[:,:num_points,:]

	#shuffle point clouds per epoch
	idx = np.arange(len(labels))
	np.random.shuffle(idx)

	sampled = sampled[idx]
	labels = labels[idx]

	return sampled, labels

def get_current_data_withmask_h5(pcs, labels, masks, num_points, shuffle=True):
	#shuffle points to sample
	idx_pts = np.arange(pcs.shape[1])

	if (shuffle):
		# print("Shuffled points: "+str(shuffle))
		np.random.shuffle(idx_pts)

	sampled = pcs[:,idx_pts[:num_points],:]
	sampled_mask = masks[:,idx_pts[:num_points]]

	#shuffle point clouds per epoch
	idx = np.arange(len(labels))

	##Shuffle order of the inputs
	if (shuffle):
		np.random.shuffle(idx)

	sampled = sampled[idx]
	sampled_mask = sampled_mask[idx]
	labels = labels[idx]

	return sampled, labels, sampled_mask

def get_current_data_parts_h5(pcs, labels, parts, num_points):
	#shuffle points to sample
	idx_pts = np.arange(pcs.shape[1])
	np.random.shuffle(idx_pts)

	sampled = pcs[:,idx_pts[:num_points],:]

	sampled_parts = parts[:,idx_pts[:num_points]]

	#shuffle point clouds per epoch
	idx = np.arange(len(labels))
	np.random.shuffle(idx)

	sampled = sampled[idx]
	sampled_parts = sampled_parts[idx]
	labels = labels[idx]

	return sampled, labels, sampled_parts

def get_current_data_discriminator_h5(pcs, labels, types, num_points):
	#shuffle points to sample
	idx_pts = np.arange(pcs.shape[1])
	np.random.shuffle(idx_pts)

	sampled = pcs[:,idx_pts[:num_points],:]

	#shuffle point clouds per epoch
	idx = np.arange(len(labels))
	np.random.shuffle(idx)

	sampled = sampled[idx]
	sampled_types = types[idx]
	labels = labels[idx]

	return sampled, labels, sampled_types


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return data, label

def load_withmask_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    mask = f['mask'][:]

    return data, label, mask

def load_discriminator_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    model_type = f['type'][:]

    return data, label, model_type

def load_parts_h5(h5_filename):
	f = h5py.File(h5_filename)
	data = f['data'][:]
	label = f['label'][:]
	parts = f['parts'][:]

	return data, label, parts


def convert_to_binary_mask(masks):
	binary_masks = []
	for i in range(masks.shape[0]):
		binary_mask = np.ones(masks[i].shape)
		bg_idx = np.where(masks[i,:]==-1)
		binary_mask[bg_idx] = 0

		binary_masks.append(binary_mask)
	
	binary_masks = np.array(binary_masks)
	return binary_masks		

def flip_types(types):
	types = (types==0)
	return types	

