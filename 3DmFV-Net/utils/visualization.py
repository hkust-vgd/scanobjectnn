import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import sklearn.metrics
import itertools
import os
import sys
import pickle
import tensorflow as tf

import provider
import utils
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils/'))
import pc_util
import tf_util
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.mplot3d import proj3d

def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,0,zback]])


def draw_point_cloud(points, output_filename='default_output_name', display=False, ax='none', color='b', vmin=0, vmax=1):
    """ points is a Nx3 numpy array """
    if ax=='none':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], marker='.', color=color, vmin=vmin,vmax=vmax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim([-1,1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    #savefig(output_filename)
    if display:
        plt.show()

    return ax


def draw_gaussians(gmm, ax='none',display=False, mappables=None, thresh=0):
    # gmm.p.weights_, gmm.p.means_, gmm.p.covars_
    if mappables is None:
        mappables=gmm.weights_
    if ax=='none':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(0,0)
        set_ax_props(ax)
        #proj3d.persp_transformation = orthogonal_proj

    x, y, z = sphere(subdev=20)
    n_gaussians = len(gmm.weights_)
    for i in range(n_gaussians):
        X = x*np.sqrt(gmm.covariances_[i][0]) + gmm.means_[i][0]
        Y = y*np.sqrt(gmm.covariances_[i][1]) + gmm.means_[i][1]
        Z = z*np.sqrt(gmm.covariances_[i][2]) + gmm.means_[i][2]
        cmap = cm.ScalarMappable()
        cmap.set_cmap('jet')
        cmap.set_clim(np.min(mappables),np.max(mappables))
        c = cmap.to_rgba( mappables[i])
        if mappables[i] > thresh:
            ax.plot_surface(X, Y, Z, color=c, alpha=0.3, linewidth=1)

    if display:
        plt.show()
    return ax

def draw_gaussian_points(points, g_points, gmm, idx=1, ax=None, display=False, color_val = 0, title=None, vmin=-1,vmax=1, colormap_type='jet'):
    if g_points.size==0:
        print('No points in this gaussian forthe given threshold...')
        return None
    if ax==None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = set_ax_props(ax)
    if title is not None:
        ax.set_title(title)


    x, y, z = sphere()
    n_gaussians = len(gmm.weights_)

    X = x*np.sqrt(gmm.covariances_[idx][0]) + gmm.means_[idx][0]
    Y = y*np.sqrt(gmm.covariances_[idx][1]) + gmm.means_[idx][1]
    Z = z*np.sqrt(gmm.covariances_[idx][2]) + gmm.means_[idx][2]

    ax.plot_surface(X, Y, Z, alpha=0.4, linewidth=1)


    cmap = cm.ScalarMappable()
    cmap.set_cmap(colormap_type)
    cmap.set_clim(vmin, vmax)
    c = cmap.to_rgba(color_val)

    #ax = draw_point_cloud(points, ax=ax)
    ax = draw_point_cloud(g_points, points, ax=ax, color=c, vmin=vmin, vmax=vmax)



    if display: plt.show()
    return ax


def visualize_fv(fv, gmm, label_title='none', max_n_images=5, normalization=True, export=False, display=False, filename='fisher_vectors',n_scales=1, type='generic', fig_title='Figure'):
    """ visualizes the fisher vector representation as an image
    INPUT: fv - n_gaussians*7 / B x n_gaussians*7 - fisher vector representation
           gmm.p - sklearn GaussianMixture object containing the information about the gmm.p that created the fv
           label_title - list of string labels for each model
            max_n_images - scalar int limiting the number of images toplot
    OUTPUT: None (opens a window and draws the axes)
    """
    cmap = "seismic"
    scalefactor= 1 if normalization==True else 0.05
    vmin = -1 * scalefactor
    vmax = 1 * scalefactor

    n_gaussians = len(gmm.means_)

    if type == 'generic':
        derivatives = ["d_pi", "d_mu1", "d_mu2", "d_mu3", "d_sig1", "d_sig2", "d_sig3"]
    elif type == 'minmax':
        derivatives = ["d_pi_max","d_pi_sum",
                       "d_mu1_max", "d_mu2_max", "d_mu3_max",
                       "d_mu1_min", "d_mu2_min", "d_mu3_min",
                       "d_mu1_sum", "d_mu2_sum", "d_mu3_sum",
                       "d_sig1_max", "d_sig2_max", "d_sig3_max",
                       "d_sig1_min", "d_sig2_min", "d_sig3_min",
                       "d_sig1_sum", "d_sig2_sum", "d_sig3_sum"]
    else:
        derivatives=[]

    tick_marks = np.arange(len(derivatives))

    if len(fv.shape) == 1:
        # #Single fv
        # d_pi = np.expand_dims(fv[0][0:n_gaussians],axis=0)
        # d_mu = np.reshape(fv[0][n_gaussians:n_gaussians*(n_features+fv_noise)],[n_features, n_gaussians])
        # d_sigma = np.reshape(fv[0][n_gaussians*(n_features+fv_noise):n_gaussians*(n_covariances + n_features+fv_noise)],[n_covariances, n_gaussians])
        # fv_mat = np.concatenate([d_pi,d_mu,d_sigma], axis=0)
        fig = plt.figure()
        fv_mat = np.reshape(fv,(-1,int(np.round(n_gaussians/n_scales))))
        plt.imshow(fv_mat, cmap=cmap, vmin=vmin, vmax=vmax)
        ax = plt.gca()
        ax.set_title(label_title)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(derivatives)

    else:
        #Batch fv
        n_models = fv.shape[0]
        if n_models > max_n_images:
            n_models = max_n_images #Limit the number of images
        f, ax = plt.subplots(n_models, squeeze=False)
        f.canvas.set_window_title(fig_title)
        for i in range(n_models):

            if len(fv.shape) == 2:
                # flattened
                fv_mat = np.reshape(fv[i,:], (-1, int(np.round(n_gaussians/n_scales))))
            else:
                fv_mat = fv[i,:,:]

            ax[i, 0].imshow(fv_mat, cmap=cmap, vmin=vmin,vmax=vmax)
            ax[i, 0].set_title(label_title[i])
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])
            #ax[i, 0].axis('off')
            ax[i, 0].set_yticks(tick_marks)
            ax[i, 0].set_yticklabels(derivatives)
            ax[i, 0].tick_params(labelsize=3)


        plt.subplots_adjust(hspace=0.5)

    if export:
        plt.savefig(filename + '.pdf',format='pdf', bbox_inches='tight', dpi=1000)
    if display:
        plt.show()

def visualize_pc_seg(points, seg, color_map, label_title=None, fig_title='figure', export=False, filename='seg', format='png' ):
    """ visualizes the point cloud  with color coded segmentation as an image
    INPUT: points - XYZ coordinates BXn_pointsx3
            seg - color coded segmentation
    OUTPUT: None - exports the image to a file
    """
    n_colors = len(color_map)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = provider.rotate_x_point_cloud_by_angle(points, -0.5*np.pi)
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', color_map, N=n_colors)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=seg, cmap=mycmap, marker='.', vmin=0, vmax=n_colors, edgecolors='none')
    ax.view_init(elev=35.264, azim=45)
    axisEqual3D(ax)
    ax.axis('off')
    # plt.show()

    if export:
        if format=='png':
            plt.savefig(filename + '.png', format='png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

def visualize_pc_seg_diff(points, seg_gt, seg_pred, color_map, label_title=None, fig_title='figure', export=False, filename='seg', format='png' ):
    """ visualizes the point cloud  with red and blut color coding the difference of the prediction from the ground truth
    INPUT:
    OUTPUT:
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = provider.rotate_x_point_cloud_by_angle(points, -0.5*np.pi)
    mycmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', [[1.0, 0.0, 0.0],[0.0, 0.0, 1.0]], N=2)
    diff_idx = np.int32(seg_gt == seg_pred)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=diff_idx, cmap=mycmap, marker='.', vmin=0, vmax=1, edgecolors='none')
    ax.view_init(elev=35.264, azim=45)
    axisEqual3D(ax)
    ax.axis('off')
    # plt.show()

    if export:
        if format=='png':
            plt.savefig(filename + '.png', format='png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

def make_segmentation_triplets_for_paper(path, cls='Chair', export=False):

    image_types = ['/gt/', '/pred/', '/diff/']
    output_dir = path + '/triplet_images'

    if cls == 'all':
        hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
        all_obj_cat_file = os.path.join(hdf5_data_dir, 'all_object_categories.txt')
        fin = open(all_obj_cat_file, 'r')
        lines = [line.rstrip() for line in fin.readlines()]
        objnames = [line.split()[0] for line in lines]
        n_objects = len(objnames)
        filename = output_dir + '/' + 'all'
    else:
        n_objects = 1
        filename = output_dir + '/' + cls.title()
        objnames = [cls.title()]

    fig = plt.figure()
    ax = AxesGrid(fig, 111, nrows_ncols=(n_objects, 3), axes_pad=0.0)
    for i, obj in enumerate(objnames):
        cls_file_path = path+'/images/' + obj
        for j, img_type in enumerate(image_types):
            file_names = [os.path.join(cls_file_path + img_type, f) for f in os.listdir(cls_file_path + img_type)]
            file_names.sort()
            img = mpimg.imread(file_names[0])
            w = img.shape[1]
            h = img.shape[0]
            x0 = int(np.round(w * 0.25))
            y0 = int(np.round(h * 0.1))
            cropped_img = img[y0:y0+int(0.7*h),x0:x0+int(0.5*w),:]
            ax[3*i+j].axis('off')
            ax[3*i+j].imshow(cropped_img)

    #Visualize and export
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if export:
        plt.savefig(filename + '.png', format='png', bbox_inches='tight', dpi=600)
    else:
        plt.show()


def visualize_pc(points, label_title=None, fig_title='figure', export=False, filename='fv_pc', display=False):
    """ visualizes the point cloud representation as an image
    INPUT:
    OUTPUT:
    """

    f = plt.figure()
    ax = plt.axes()
    f.canvas.set_window_title(fig_title)

    # plt.get_current_fig_manager().window.wm_geometry(str(pos[0]) + "x" + str(pos[128]) + "+"+str(pos[256])+"+"+str(pos[512]))

    image = pc_util.point_cloud_isoview(points[0,:,:])
    image = np.ma.masked_where(image < 0.0005, image)
    cmap = plt.cm.rainbow
    cmap.set_bad(color='white')

    ax.imshow(image, cmap=cmap)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(label_title)
    #ax.axis('off')


    if export:
        plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight', dpi=1000)
    if display:
        plt.show()


def visualize_fv_with_pc(fv, points, label_title=None, fig_title='figure', type='minmax', pos=[750,800,0,0], export=False, filename='fv_pc'):
    """ visualizes the fisher vector representation as an image
    INPUT: fv - B X n_gaussians X n_components - fisher vector representation
            points B X n_points X 64
    OUTPUT: None (opens a window and draws the axes)
    """

    n_models = fv.shape[0]
    scalefactor = 1
    vmin = -1 * scalefactor
    vmax = 1 * scalefactor

    if type == 'generic':
        derivatives = ["d_pi", "d_mu1", "d_mu2", "d_mu3", "d_sig1", "d_sig2", "d_sig3"]
    elif type == 'minmax':
        derivatives = ["d_pi_max","d_pi_sum",
                       "d_mu1_max", "d_mu2_max", "d_mu3_max",
                       "d_mu1_min", "d_mu2_min", "d_mu3_min",
                       "d_mu1_sum", "d_mu2_sum", "d_mu3_sum",
                       "d_sig1_max", "d_sig2_max", "d_sig3_max",
                       "d_sig1_min", "d_sig2_min", "d_sig3_min",
                       "d_sig1_sum", "d_sig2_sum", "d_sig3_sum"]
    else:
        derivatives = []
    tick_marks = np.arange(len(derivatives))

    f, ax = plt.subplots(n_models, 2, squeeze=False)
    f.canvas.set_window_title(fig_title)

    plt.get_current_fig_manager().window.wm_geometry(str(pos[0]) + "x" + str(pos[1]) + "+"+str(pos[2])+"+"+str(pos[3]))

    for i in range(n_models):
        cmap = "seismic"
        ax[i, 0].imshow(fv[i, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        ax[i, 0].set_title(label_title[i])
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        # ax[i, 0].axis('off')
        ax[i, 0].set_yticks(tick_marks)
        ax[i, 0].set_yticklabels(derivatives)
        ax[i, 0].tick_params(labelsize=3)

        #image = pc_util.point_cloud_three_views(points[i, :, :])
        image = pc_util.point_cloud_isoview(points[i, :, :])
        image = np.ma.masked_where(image < 0.0005, image)
        cmap = plt.cm.rainbow
        cmap.set_bad(color='white')


        ax[i, 1].imshow(image, cmap=cmap)
        ax[i, 1].get_xaxis().set_visible(False)
        ax[i, 1].get_yaxis().set_visible(False)


    if export:
        plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight', dpi=1000)

def visualize_single_fv_with_pc(fv, points, label_title=None, fig_title='figure', type='minmax', pos=[750,800,0,0], export=False, filename='fv_pc'):
    """ visualizes the fisher vector representation as an image
    INPUT: fv - B X n_gaussians X n_components - fisher vector representation
            points B X n_points X 64
    OUTPUT: None (opens a window and draws the axes)
    """

    n_models = fv.shape[0]
    cmap = "seismic"
    scalefactor = 1
    vmin = -1 * scalefactor
    vmax = 1 * scalefactor

    if type == 'generic':
        derivatives = ["d_pi", "d_mu1", "d_mu2", "d_mu3", "d_sig1", "d_sig2", "d_sig3"]
    elif type == 'minmax':
        derivatives = ["d_pi_max","d_pi_sum",
                       "d_mu1_max", "d_mu2_max", "d_mu3_max",
                       "d_mu1_min", "d_mu2_min", "d_mu3_min",
                       "d_mu1_sum", "d_mu2_sum", "d_mu3_sum",
                       "d_sig1_max", "d_sig2_max", "d_sig3_max",
                       "d_sig1_min", "d_sig2_min", "d_sig3_min",
                       "d_sig1_sum", "d_sig2_sum", "d_sig3_sum"]
    else:
        derivatives = []
    tick_marks = np.arange(len(derivatives))

    f = plt.figure()
    f.canvas.set_window_title(fig_title)
    ax1 = plt.axes([0.05, 0.5, 0.45, 0.2])
    ax2 = plt.axes([0.5, 0.5, 0.3, 0.3])
    #plt.get_current_fig_manager().window.wm_geometry(str(pos[0]) + "x" + str(pos[128]) + "+"+str(pos[256])+"+"+str(pos[512]))

    #for i in range(n_models):
    ax1.imshow(fv[0,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_title(label_title)
    ax1.set_xticks([])
    ax1.set_yticks([])
    # ax[i, 0].axis('off')
    ax2.set_yticks(tick_marks)
    ax2.set_yticklabels(derivatives)
    ax2.tick_params(labelsize=3)

    #image = pc_util.point_cloud_three_views(points[i, :, :])
    image = pc_util.point_cloud_isoview(points[0,:,:])
    image = np.ma.masked_where(image < 0.0005, image)
    cmap = plt.cm.rainbow
    cmap.set_bad(color='white')

    ax2.imshow(image, cmap=cmap)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    #fig.patch.set_visible(False)
    ax2.axis('off')


    if export:
        plt.savefig(filename + '.pdf', format='pdf', bbox_inches='tight', dpi=1000)

def visualize_confusion_matrix(y_true, y_pred, classes=None, normalize=False, cmap=cm.jet, export=False, display=False, filename='confusion_mat', n_classes=40):
    """
    plots the confusion matrix as and image
    :param y_true: list of the GT label of the models
    :param y_pred: List of the predicted label of the models
    :param classes: List of strings containing the label tags
    :param normalize: bool indicating if to normalize the confusion matrix
    :param cmap: colormap to use for plotting
    :return: None (just plots)
    """
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=range(0,n_classes))
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    fig = plt.figure()
    plt.imshow(conf_mat, cmap=cmap)
    ax = plt.gca()
    ax.set_title('Confusion Matrix')

    #Write the labels for each row and column
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90,  fontsize=5)
        plt.yticks(tick_marks, classes,  fontsize=5)

    #Write the values in the center of the cell
    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, conf_mat[i, j],
                 horizontalalignment="center", fontsize=3,
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if export:
        plt.savefig(filename +'.pdf',format='pdf', bbox_inches='tight', dpi=1000)
    if display:
        plt.show()


def sphere(subdev=10):
    #helper function to compute the coordinates of a unit sphere centered at 0,0,0
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return (x,y,z)


def set_ax_props(ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    return ax


def visualize_derivatives(points, gmm, gaussian_index,per_point_d_pi, per_point_d_mu, per_point_d_sigma):


    fig = plt.figure()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    ax1 = fig.add_subplot(131, projection='3d')
    ax1 = set_ax_props(ax1)
    ax1.view_init(0, 90)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2 = set_ax_props(ax2)
    ax2.view_init(0, 0)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3 = set_ax_props(ax3)
    ax3.view_init(0, 0)

    point_d_mux = per_point_d_mu[:, gaussian_index, 0]
    point_d_muy = per_point_d_mu[:, gaussian_index, 1]
    point_d_muz = per_point_d_mu[:, gaussian_index, 2]

    d_mu_range = [-1,1]
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax1, display=False, color_val=point_d_mux,
                         title='mu_x',vmin=d_mu_range[0], vmax=d_mu_range[1], colormap_type='seismic')
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax2, display=False, color_val=point_d_muy,
                         title='mu_y',vmin=d_mu_range[0], vmax=d_mu_range[1], colormap_type='seismic')
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax3, display=False, color_val=point_d_muz,
                         title='mu_z',vmin=d_mu_range[0], vmax=d_mu_range[1], colormap_type='seismic')

    fig = plt.figure()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    ax1 = fig.add_subplot(131, projection='3d')
    ax1 = set_ax_props(ax1)
    ax1.view_init(0, 90)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2 = set_ax_props(ax2)
    ax2.view_init(0, 0)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3 = set_ax_props(ax3)
    ax3.view_init(0, 0)

    point_d_sigx = per_point_d_sigma[:, gaussian_index, 0]
    point_d_sigy = per_point_d_sigma[:, gaussian_index, 1]
    point_d_sigz = per_point_d_sigma[:, gaussian_index, 2]

    d_sig_range = [-1, 1]
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax1, display=False, color_val=point_d_sigx,
                         title='sig_x',vmin=d_sig_range[0], vmax=d_sig_range[1], colormap_type='seismic')
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax2, display=False, color_val=point_d_sigy,
                         title='sig_y',vmin=d_sig_range[0], vmax=d_sig_range[1], colormap_type='seismic')
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax3, display=False, color_val=point_d_sigz,
                         title='sig_z',vmin=d_sig_range[0], vmax=d_sig_range[1], colormap_type='seismic')

    fig = plt.figure()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    d_pi_range = [-1, 1]
    ax_pi = fig.add_subplot(111, projection='3d')
    ax_pi = set_ax_props(ax_pi)
    draw_gaussian_points(points, points, gmm, idx=gaussian_index, ax=ax_pi, display=False, color_val=per_point_d_pi[:, gaussian_index],
                         title='d_pi',vmin=d_pi_range[0], vmax=d_pi_range[1], colormap_type='seismic')

    plt.show()


def visualize_fv_pc_clas():
    num_points = 1024
    n_classes = 40
    clas = 'person'
    #Create new gaussian
    subdev = 5
    variance = 0.04
    export = False
    display = True
    exp_path = '/home/itzikbs/PycharmProjects/fisherpointnet/paper_images/'

    shape_names = provider.getDataFiles( \
        os.path.join(BASE_DIR, 'data/modelnet' + str(n_classes) + '_ply_hdf5_2048/shape_names.txt'))
    shape_dict = {shape_names[i]: i for i in range(len(shape_names))}

    gmm = utils.get_grid_gmm(subdivisions=[subdev, subdev, subdev], variance=variance)
    # compute fv
    w = tf.constant(gmm.weights_, dtype=tf.float32)
    mu = tf.constant(gmm.means_, dtype=tf.float32)
    sigma = tf.constant(gmm.covariances_, dtype=tf.float32)

    for clas in shape_dict:
        points = provider.load_single_model_class(clas=clas, ind=0, test_train='train', file_idxs=0, num_points=1024,
                                                  n_classes=n_classes)
        points = np.expand_dims(points,0)

        points_tensor = tf.constant(points, dtype=tf.float32)  # convert points into a tensor
        fv_tensor = tf_util.get_fv_minmax(points_tensor, w, mu, sigma, flatten=False)

        sess = tf_util.get_session(2)
        with sess:
            fv = fv_tensor.eval()
        #
        # visualize_single_fv_with_pc(fv_train, points, label_title=clas,
        #                      fig_title='fv_pc', type='paper', pos=[750, 800, 0, 0], export=export,
        #                      filename=BASE_DIR + '/paper_images/fv_pc_' + clas)

        visualize_fv(fv, gmm, label_title=[clas], max_n_images=5, normalization=True, export=export, display=display,
                     filename=exp_path + clas+'_fv', n_scales=1, type='none', fig_title='Figure')
        visualize_pc(points, label_title=clas, fig_title='figure', export=export, filename=exp_path +clas+'_pc')
        plt.close('all')

    #plt.show()

def visualize_pc_with_svd():
    num_points = 1024
    model_idx = 5
    gpu_idx = 0

    original_points, _ = provider.load_single_model(model_idx=model_idx, test_train='train', file_idxs=0, num_points=num_points)
    original_points = provider.rotate_point_cloud_by_angle(original_points, np.pi/2)

    # #Simple plane sanity check
    # original_points = np.concatenate([np.random.rand(2, 1024), np.zeros([1, 1024])],axis=0)
    # R = np.array([[0.7071, 0, 0.7071],
    #               [0, 1, 0],
    #               [-0.7071, 0, 0.7071]])
    # original_points = np.transpose(np.dot(R ,original_points))

    original_points = np.expand_dims(original_points,0)
    pc_util.pyplot_draw_point_cloud(original_points[0,:,:])

    sess = tf_util.get_session(gpu_idx, limit_gpu=True)
    points_pl = tf.placeholder(tf.float32, shape=(1, num_points, 3))
    svd_op = tf_util.pc_svd(points_pl)
    rotated_points = sess.run(svd_op, feed_dict={points_pl:original_points})

    pc_util.pyplot_draw_point_cloud(rotated_points[0,:,:])
    plt.show()

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR+'/visualization')
    log_dir = 'log_fisher_grid5_nonlinear'

    #Load gaussians
    # gmm_filename = os.path.join(log_dir,'gmm.p')
    # gmm = pickle.load(open(gmm_filename, "rb"))
    # parameters_filename =  os.path.join(log_dir,'parameters.p')
    # PARAMETERS = pickle.load(open(parameters_filename, "rb"))

    #Create new gaussian
    subdev = 10
    variance = 0.01
    gmm = utils.get_grid_gmm(subdivisions=[subdev, subdev, subdev], variance=variance)

    class helper_struct():
        def __init__(self):
            self.num_gaussians = subdev
            self.gmm_type = 'grid'

    PARAMETERS = helper_struct()

    gaussian_index = 740
    num_points = 1024
    model_idx = 0
    n_gaussians = np.power(PARAMETERS.num_gaussians, 3) if PARAMETERS.gmm_type == 'grid' else PARAMETERS.num_gaussians
    points,_ = provider.load_single_model(model_idx = model_idx,test_train = 'train', file_idxs=0, num_points = num_points)

    g_pts, g_probs = utils.get_gaussian_points(points, gmm, idx=gaussian_index, thresh=0.01)
    #draw_gaussian_points(points, g_pts, gmm, idx=gaussian_index, ax=None, display=True, color_val=g_probs)

    #fv = utils.fisher_vector(points, gmm, normalization=True)
    # d_pi = fv[0:n_gaussians]
    # mean_d_pi = 0.02
    # ax=draw_point_cloud(points)
    # draw_gaussians(gmm, ax=ax, display=True, mappables=d_pi, thresh=mean_d_pi)

    per_point_dpi,per_point_d_mu, per_point_d_sigma = utils.fisher_vector_per_point( points, gmm)
    visualize_derivatives(points, gmm,gaussian_index,per_point_dpi, per_point_d_mu, per_point_d_sigma)



if __name__ == "__main__":
    #main()
    visualize_fv_pc_clas()
    # path_to_test_results = '/home/itzikbs/PycharmProjects/fisherpointnet/log_seg/test_results'
    # make_segmentation_triplets_for_paper(path_to_test_results, cls='all', export = True)
    # visualize_pc_with_svd()