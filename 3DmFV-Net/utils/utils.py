from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
import os
import pickle
import numpy as np

import provider


def get_gmm(points, n_gaussians, NUM_POINT, type='grid', variance=0.05, n_scales=3, D=3):
    """
    Compute weights, means and covariances for a gmm with two possible types 'grid' (2D/3D) and 'learned'

    :param points: num_points_per_model*nummodels X 3 - xyz coordinates
    :param n_gaussians: scalar of number of gaussians /  number of subdivisions for grid type
    :param NUM_POINT: number of points per model
    :param type: 'grid' / 'leared' toggle between gmm methods
    :param variance: gaussian variance for grid type gmm
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    if type == 'grid':
        #Generate gaussians on a grid - supports only axis equal subdivisions
        if n_gaussians >= 32:
            print('Warning: You have set a very large number of subdivisions.')
        if not(isinstance(n_gaussians, list)):
            if D == 2:
                gmm = get_2d_grid_gmm(subdivisions=[n_gaussians, n_gaussians], variance=variance)
            elif D == 3:
                gmm = get_3d_grid_gmm(subdivisions=[n_gaussians, n_gaussians, n_gaussians], variance=variance)
            else:
                ValueError('Wrong dimension. This supports either D=2 or D=3')

    elif type == 'learn':
        #learn the gmm from given data and save to gmm.p file, if already learned then load it from existing gmm.p file for speed
        if isinstance(n_gaussians, list):
            raise  ValueError('Wrong number of gaussians: non-grid value must be a scalar')
        print("Computing GMM from data - this may take a while...")
        info_str = "g" + str(n_gaussians) + "_N" + str(len(points)) + "_M" + str(len(points) / NUM_POINT)
        gmm_dir = "gmms"
        if not os.path.exists(gmm_dir):
            os.mkdir(gmm_dir)
        filename = gmm_dir + "/gmm_" + info_str + ".p"
        if os.path.isfile(filename):
            gmm = pickle.load(open(filename, "rb"))
        else:
            gmm = get_learned_gmm(points, n_gaussians, covariance_type='diag')
            pickle.dump(gmm, open( filename, "wb"))
    else:
        ValueError('Wrong type of GMM [grid/learn]')

    return gmm


def get_learned_gmm(points, n_gaussians, covariance_type='diag'):
    """
    Learn weights, means and covariances for a gmm based on input data using sklearn EM algorithm

    :param points: num_points_per_model*nummodels X 3 - xyz coordinates
    :param n_gaussians: scalar of number of gaussians /  3 element list of number of subdivisions for grid type
    :param covariance_type: Specify the type of covariance mmatrix : 'diag', 'full','tied', 'spherical' (Note that the Fisher Vector method relies on diagonal covariance matrix)
        See sklearn documentation : http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    gmm = GaussianMixture(n_components = n_gaussians, covariance_type=covariance_type)
    gmm.fit(points.astype(np.float64))
    return gmm


def get_3d_grid_gmm(subdivisions=[5,5,5], variance=0.04):
    """
    Compute the weight, mean and covariance of a gmm placed on a 3D grid
    :param subdivisions: 2 element list of number of subdivisions of the 3D space in each axes to form the grid
    :param variance: scalar for spherical gmm.p
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [1.0/(subdivisions[0]),  1.0/(subdivisions[1]),  1.0/(subdivisions[2])]

    means = np.mgrid[ step[0]-1: 1.0-step[0]: complex(0, subdivisions[0]),
                      step[1]-1: 1.0-step[1]: complex(0, subdivisions[1]),
                      step[2]-1: 1.0-step[2]: complex(0, subdivisions[2])]
    means = np.reshape(means, [3, -1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


def get_2d_grid_gmm(subdivisions=[5, 5], variance=0.04):
    """
    Compute the weight, mean and covariance of a 2D gmm placed on a 2D grid

    :param subdivisions: 2 element list of number of subdivisions of the 2D space in each axes to form the grid
    :param variance: scalar for spherical gmm.p
    :return gmm: gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    """
    # n_gaussians = reduce(lambda x, y: x*y,subdivisions)
    n_gaussians = np.prod(np.array(subdivisions))
    step = [1.0/(subdivisions[0]),  1.0/(subdivisions[1])]

    means = np.mgrid[step[0]-1: 1.0-step[0]: complex(0, subdivisions[0]),
            step[1]-1: 1.0-step[1]: complex(0, subdivisions[1])]
    means = np.reshape(means, [2,-1]).T
    covariances = variance*np.ones_like(means)
    weights = (1.0/n_gaussians)*np.ones(n_gaussians)
    gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')
    gmm.weights_ = weights
    gmm.covariances_ = covariances
    gmm.means_ = means
    from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
    gmm.precisions_cholesky_ = _compute_precision_cholesky(covariances, 'diag')
    return gmm


def get_fisher_vectors(points,gmm, normalization=True):
    """
    Compute the fisher vector representation of a point cloud or a batch of point clouds

    :param points: n_points x 3 / B x n_points x 3
    :param gmm: sklearn MixtureModel class containing the gmm.p parameters.p
    :return: fisher vector representation for a single point cloud or a batch of point clouds
    """

    if len(points.shape) == 2:
        # single point cloud
        fv = fisher_vector(points, gmm, normalization=normalization)
    else:
        # Batch of  point clouds
        fv = []
        n_models = points.shape[0]
        for i in range(n_models):
            fv.append(fisher_vector(points[i], gmm, normalization=True))
        fv = np.array(fv)
    return fv


def fisher_vector(xx, gmm, normalization=True):
    """
    Computes the Fisher vector on a set of descriptors.
    code from : https://gist.github.cnsom/danoneata/9927923
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors

    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.

    Returns
    -------
    fv: array_like, shape (K + 128 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.

    Reference
    ---------
    Sanchez, J., Perronnin, F., Mensink, T., & Verbeek, J. (2013).
    Image classification with the fisher vector: Theory and practice. International journal of computer vision, 105(64), 222-245.
    https://hal.inria.fr/hal-00830491/file/journal.pdf

    """
    xx = np.atleast_2d(xx)
    n_points = xx.shape[0]
    D = gmm.means_.shape[1]
    tiled_weights = np.tile(np.expand_dims(gmm.weights_, axis=-1), [1, D])

    #start = time.time()
    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK
    #mid = time.time()
    #print("Computing the probabilities took ", str(mid-start))
    #Compute Derivatives

    # Compute the sufficient statistics of descriptors.
    s0 = np.sum(Q, 0)[:, np.newaxis] / n_points
    s1 = np.dot(Q.T, xx) / n_points
    s2 = np.dot(Q.T, xx ** 2) / n_points

    d_pi = (s0.squeeze() - n_points * gmm.weights_) / np.sqrt(gmm.weights_)
    d_mu = (s1 - gmm.means_ * s0 ) / np.sqrt(tiled_weights*gmm.covariances_)
    d_sigma = (
        + s2
        - 2 * s1 * gmm.means_
        + s0 * gmm.means_ ** 2
        - s0 * gmm.covariances_
        ) / (np.sqrt(2*tiled_weights)*gmm.covariances_)

    #Power normaliation
    alpha = 0.5
    d_pi = np.sign(d_pi) * np.power(np.absolute(d_pi),alpha)
    d_mu = np.sign(d_mu) * np.power(np.absolute(d_mu), alpha)
    d_sigma = np.sign(d_sigma) * np.power(np.absolute(d_sigma), alpha)

    if normalization == True:
        d_pi = normalize(d_pi[:,np.newaxis], axis=0).ravel()
        d_mu = normalize(d_mu, axis=0)
        d_sigma = normalize(d_sigma, axis=0)
    # Merge derivatives into a vector.

    #print("comnputing the derivatives took ", str(time.time()-mid))

    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


def fisher_vector_per_point( xx, gmm):
    """
    see notes for above function - performs operations per point

    :param xx: array_like, shape (N, D) or (D, )- The set of descriptors
    :param gmm: instance of sklearn mixture.GMM object - Gauassian mixture model of the descriptors.
    :return: fv_per_point : fisher vector per point (derivative by w, derivative by mu, derivative by sigma)
    """
    xx = np.atleast_2d(xx)
    n_points = xx.shape[0]
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]

    sig2 = np.array([gmm.covariances_.T[0, :], gmm.covariances_.T[1, :], gmm.covariances_.T[2,:]]).T
    sig2_tiled = np.tile(np.expand_dims(sig2, axis=0), [n_points, 1, 1])

    # Compute derivativees per point and then sum.
    Q = gmm.predict_proba(xx)  # NxK
    tiled_weights = np.tile(np.expand_dims(gmm.weights_, axis=-1), [1, D])
    sqrt_w = np.sqrt(tiled_weights)

    d_pi = (Q - np.tile(np.expand_dims(gmm.weights_, 0), [n_points, 1])) / np.sqrt(np.tile(np.expand_dims(gmm.weights_, 0), [n_points, 1]))
    x_mu = np.tile( np.expand_dims(xx, axis=2), [1, 1, n_gaussians]) - np.tile(np.expand_dims(gmm.means_.T, axis=0), [n_points, 1, 1])
    x_mu = np.swapaxes(x_mu, 1, 2)
    d_mu = (np.tile(np.expand_dims(Q, -1), D) * x_mu) / (np.sqrt(sig2_tiled) * sqrt_w)

    d_sigma =   np.tile(np.expand_dims(Q, -1), 3)*((np.power(x_mu,2)/sig2_tiled)-1)/(np.sqrt(2)*sqrt_w)

    fv_per_point = (d_pi, d_mu, d_sigma)
    return fv_per_point


def l2_normalize(v, dim=1):
    """
    Normalize a vector along a dimension

    :param v: a vector or matrix to normalize
    :param dim: the dimension along which to normalize
    :return: normalized v along dim
    """
    norm = np.linalg.norm(v, axis=dim)
    if norm.all() == 0:
       return v
    return v / norm


def get_3DmFV(points, w, mu, sigma, normalize=True):
    """
       Compute the 3D modified fisher vectors given the gmm model parameters (w,mu,sigma) and a set of points
       For faster performance (large batches) use the tensorflow version

       :param points: B X N x 3 tensor of XYZ points
       :param w: B X n_gaussians tensor of gaussian weights
       :param mu: B X n_gaussians X 3 tensor of gaussian cetnters
       :param sigma: B X n_gaussians X 3 tensor of stddev of diagonal covariance
       :return: fv: B X 20*n_gaussians tensor of the fisher vector
       """
    n_batches = points.shape[0]
    n_points = points.shape[1]
    n_gaussians = mu.shape[0]
    D = mu.shape[1]

    # Expand dimension for batch compatibility
    batch_sig = np.tile(np.expand_dims(sigma, 0), [n_points, 1, 1])  # n_points X n_gaussians X D
    batch_sig = np.tile(np.expand_dims(batch_sig, 0), [n_batches, 1, 1, 1])  # n_batches X n_points X n_gaussians X D
    batch_mu = np.tile(np.expand_dims(mu, 0), [n_points, 1, 1])  # n_points X n_gaussians X D
    batch_mu = np.tile(np.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1])  # n_batches X n_points X n_gaussians X D
    batch_w = np.tile(np.expand_dims(np.expand_dims(w, 0), 0), [n_batches, n_points,
                                                                1])  # n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = np.tile(np.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1])  # n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    # Compute derivatives
    w_per_batch_per_d = np.tile(np.expand_dims(np.expand_dims(w, 0), -1),
                                [n_batches, 1, 3*D])  # n_batches X n_gaussians X 3*D (D for min and D for max)

    # Define multivariate noraml distributions
    # Compute probability per point
    p_per_point = (1.0 / (np.power(2.0 * np.pi, D / 2.0) * np.power(batch_sig[:, :, :, 0], D))) * np.exp(
        -0.5 * np.sum(np.square((batch_points - batch_mu) / batch_sig), axis=3))

    w_p = p_per_point
    Q = w_p  # enforcing the assumption that the sum is 1
    Q_per_d = np.tile(np.expand_dims(Q, -1), [1, 1, 1, D])

    d_pi_all = np.expand_dims((Q - batch_w) / (np.sqrt(batch_w)), -1)
    d_pi = np.concatenate([np.max(d_pi_all, axis=1), np.sum(d_pi_all, axis=1)], axis=2)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / (np.sqrt(w_per_batch_per_d))) * np.concatenate([np.max(d_mu_all, axis=1), np.min(d_mu_all, axis=1), np.sum(d_mu_all, axis=1)], axis=2)

    d_sig_all = Q_per_d * (np.square((batch_points - batch_mu) / batch_sig) - 1)
    d_sigma = (1 / (np.sqrt(2 * w_per_batch_per_d))) * np.concatenate([np.max(d_sig_all, axis=1), np.min(d_sig_all, axis=1), np.sum(d_sig_all, axis=1)], axis=2)

    # number of points  normaliation
    d_pi = d_pi / n_points
    d_mu = d_mu / n_points
    d_sigma =d_sigma / n_points

    if normalize:
        # Power normaliation
        alpha = 0.5
        d_pi = np.sign(d_pi) * np.power(np.abs(d_pi), alpha)
        d_mu = np.sign(d_mu) * np.power(np.abs(d_mu), alpha)
        d_sigma = np.sign(d_sigma) * np.power(np.abs(d_sigma), alpha)

        # L2 normaliation
        d_pi = np.array([l2_normalize(d_pi[i, :, :], dim=0) for i in range(n_batches)])
        d_mu = np.array([l2_normalize(d_mu[i, :, :], dim=0) for i in range(n_batches)])
        d_sigma = np.array([l2_normalize(d_sigma[i, :, :], dim=0) for i in range(n_batches)])


    fv = np.concatenate([d_pi, d_mu, d_sigma], axis=2)
    fv = np.transpose(fv, axes=[0, 2, 1])

    return fv


if __name__ == "__main__":

    model_idx = 0
    num_points = 1024
    gmm = get_3d_grid_gmm(subdivisions=[5, 5, 5], variance=0.04)
    points = provider.load_single_model(model_idx=model_idx, train_file_idxs=0, num_points=num_points)
    points = np.tile(np.expand_dims(points, 0), [128, 1, 1])

    fv_gpu = get_fisher_vectors(points, gmm, normalization=True)



