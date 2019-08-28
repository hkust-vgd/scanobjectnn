import tensorflow as tf
import numpy as np
# from sklearn.mixture import GaussianMixture
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
import provider


def get_gmm_vars(n_gaussians, D, initialize='random', scope='gmm'):
    '''
    :param n_gaussians: number of gaussians
    :param D: data dimensionality
    :return: initialized tf variables for gmm model
    '''
    w_init = (1 / n_gaussians) * tf.ones(shape=[n_gaussians])
    w = tf.clip_by_value(tf.nn.softmax(tf.get_variable(name=scope+'w', initializer=w_init, dtype=tf.float32)),
                         clip_value_min=0.0001, clip_value_max=1.0)
    # w = w_init
    if initialize=='random':
        mu_init = tf.truncated_normal(mean=0.0, stddev=0.5, shape=(D, n_gaussians))
        sig_init = tf.truncated_normal(mean=0.2, stddev=0.099, shape=(D, n_gaussians))
    elif initialize == 'kmeans':
        #works only for 3D points for now
        w_init, mu_init, sig_init = get_kmeans_init(n_gaussians, cov_type='farthest')
    else:
        subdivision = int(np.round(np.power(n_gaussians, 1.0 / D)))
        #step = [fv_noise.0/subdivision for i in range(D)]
        step = 1.0/subdivision
        mu_init = np.mgrid[tuple(slice(step - 1, 1, step * 2) for _ in range(D))]
        # mu_init = np.mgrid[ step-fv_noise : fv_noise.0-step: complex(0,subdivision),
        #                     step-fv_noise : fv_noise.0-step: complex(0,subdivision),
        #                     step-fv_noise : fv_noise.0-step: complex(0,subdivision)]
        mu_init = np.reshape(mu_init, [D, -1]).astype(np.float32)
        sig_init = np.sqrt((1 / subdivision)) * np.ones(shape=[D, n_gaussians], dtype=np.float32)

    mu = tf.get_variable(initializer=mu_init,
                         dtype=tf.float32, name=scope+'mu')
    stdev = tf.clip_by_value(1 + tf.nn.elu(tf.get_variable(
        initializer=sig_init, dtype=tf.float32, name=scope+'cov')),
                             clip_value_min=0.001, clip_value_max=1.0)

    return w, mu, stdev


def get_kmeans_init(n_gaussians, cov_type='farthest'):
    D = 3

    # Get the training data for initialization
    # Load multiple models from the dataset
    points, labels, _, _ = provider.load_dataset( num_points=1024)
    mask = []
    for i in range(40):
        mask.append(np.squeeze(np.where(labels == i))[0:10])
    mask = np.concatenate(mask, axis=0)
    points = points[mask, :, :]
    points = provider.jitter_point_cloud(points, sigma=0.01, clip=0.05)
    points = np.concatenate(points, axis=0)

    #input function for kmeans clustering
    def input_fn():
        return tf.constant(points, dtype=tf.float32), None

    ## construct model
    kmeans = tf.contrib.learn.KMeansClustering(num_clusters=n_gaussians, relative_tolerance=0.0001)
    kmeans.fit(input_fn=input_fn)
    centers = kmeans.clusters()
    assignments = np.squeeze(list(kmeans.predict_cluster_idx(input_fn=input_fn)))

    n_points = points.shape[0]
    stdev = []
    w = []
    for i in range(n_gaussians):
        idx = np.squeeze(np.where(assignments == i))
        w.append(len(idx) / n_points)
        if cov_type == 'compute_cov':
            samples = points[idx, :].T
            stdev.append(np.sqrt(np.diag(np.cov(samples))))
        elif cov_type == 'farthest':
            d = np.sqrt(np.sum(np.power(points[idx, :] - centers[i, :], 2), axis=1))
            farthest_point_idx = np.argmax(d)
            stdev.append((np.max(d) / 3.) * np.ones(D))

    # gmm = GaussianMixture(n_components=n_gaussians, covariance_type='diag')

    return    w, centers.T, np.array(stdev, dtype=np.float32).T

def pairwise_distance_loss(mu, min_neighbor_dist=0.1):
    '''
    :param mu: tf variable of gmm means Dxn_gaussians
    :param neighbor_dist_thresh:  limit the minimal distance between neighbors
    :return: loss: loss function that penalizes if the distance between any two means is less than the thresholds
    '''
    n_gaussians = mu.shape[1].value
    mutile = tf.tile(tf.expand_dims(mu, 0), [n_gaussians, 1, 1])
    muTtile = tf.transpose(mutile, perm=[2, 1, 0])
    x = tf.reduce_sum(tf.pow(mutile - muTtile, 2), 1)
    D = tf.squeeze(tf.nn.relu(x) - tf.nn.relu(x - min_neighbor_dist))  # penalize if a distance is too small
    loss = -(tf.reduce_sum(D) / 2) / n_gaussians  # should have sqrt but it brakes it
    return loss


def sigma_loss(sigma, max_value=0.5, min_value=0.001):
    '''
    :param sigma: standart deviation tf variable
    :param high_thresh: limit the highest stdev
    :param low_thresh: limit the lowest stdev
    :return: loss function that enforces the std to be within the limits  (penalizes outside this range)
    '''
    loss = tf.reduce_mean(tf.nn.relu(-(sigma - min_value)) + tf.nn.relu(sigma - max_value))
    return loss


def get_mixture_log_probs(X, w, mu, stdev):
    '''

    :param X: points. the mixturemodel is estimated on these points
    :param w: gmm weights
    :param mu: gmm means
    :param stdev: gmm stadart deviation
    :return:
           mixture_dist: a gmm tensorflow object
           xLogProbs: log probability for x
    '''
    D = mu.shape[0].value
    n_gaussians = mu.shape[1].value

    xdist = []
    for i in range(n_gaussians):
        xdist.append(tf.contrib.distributions.MultivariateNormalDiag(loc=mu[:, i],
                                                                     scale_diag=stdev[:, i]
                                                                     # * tf.ones(shape=(D,fv_noise),dtype=tf.float32)
                                                                     , name='xDist1'))
    dist = tf.contrib.distributions.Categorical(probs=w)
    mixture_dist = tf.contrib.distributions.Mixture(cat=dist, components=xdist, allow_nan_stats=False)
    xLogProbs = mixture_dist.log_prob(X, name='xLogProbs1')
    return mixture_dist, xLogProbs


def get_gmm_loss(X, w, mu, stdev, cp=0.8, cmu=0.1, csig=0.1, cw=0.1, scope='loss'):
    mixture_dist, xLogProbs = get_mixture_log_probs(X, w, mu, stdev)
    n_gaussians = w.shape[0].value
    w_loss = tf.reduce_mean(tf.pow(w - 1 / n_gaussians, 2))
    mean_dist_loss = pairwise_distance_loss(mu)
    sig_loss = sigma_loss(stdev, max_value=0.25, min_value=0.00001)
    log_gmm_loss = - tf.reduce_logsumexp(tf.reduce_mean(xLogProbs, name=scope+'loss'))
    loss = cp * log_gmm_loss + cmu * mean_dist_loss + csig * sig_loss + cw * w_loss
    return loss


def get_fv_minmax(points, w, mu, sigma, flatten=True):
    """
    Compute the fisher vector given the gmm model parameters (w,mu,sigma) and a set of points

    :param points: B X N x 64 tensor of XYZ points
    :param w: B X n_gaussians tensor of gaussian weights
    :param mu: B X n_gaussians X 64 tensor of gaussian cetnters
    :param sigma: B X n_gaussians X 64 tensor of stddev of diagonal covariance
    :return: fv: B X 7*n_gaussians tensor of the fisher vector
    """
    n_batches = points.shape[0].value
    n_points = points.shape[1].value
    n_gaussians = mu.shape[0].value
    D = mu.shape[1].value

    #Expand dimension for batch compatibility
    batch_sig = tf.tile(tf.expand_dims(sigma,0),[n_points, 1, 1])  #n_points X n_gaussians X D
    batch_sig = tf.tile(tf.expand_dims(batch_sig, 0), [n_batches, 1, 1,1]) #n_batches X n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(mu, 0),[n_points, 1, 1]) #n_points X n_gaussians X D
    batch_mu = tf.tile(tf.expand_dims(batch_mu, 0), [n_batches, 1, 1, 1]) #n_batches X n_points X n_gaussians X D
    batch_w = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), 0), [n_batches, n_points, 1]) #n_batches X n_points X n_guassians X D  - should check what happens when weights change
    batch_points = tf.tile(tf.expand_dims(points, -2), [1, 1, n_gaussians,
                                                        1]) #n_batchesXn_pointsXn_gaussians_D  # Generating the number of points for each gaussian for separate computation

    #Compute derivatives
    w_per_batch = tf.tile(tf.expand_dims(w,0),[n_batches, 1]) #n_batches X n_gaussians
    w_per_batch_per_d = tf.tile(tf.expand_dims(tf.expand_dims(w, 0), -1), [n_batches, 1, 3*D]) #n_batches X n_gaussians X 128*D (D for min and D for max)
    sigma_per_batch = tf.tile(tf.expand_dims(sigma,0),[n_batches, 1, 1])
    cov_per_batch = sigma_per_batch ** 2
    mu_per_batch = tf.tile(tf.expand_dims(mu, 0),[n_batches, 1, 1])

    #Define multivariate noraml distributions
    mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=batch_mu, scale_diag=batch_sig)
    #Compute probability per point
    p_per_point = tf.exp(mvn.log_prob(batch_points))
    w_p = tf.multiply(p_per_point,batch_w)
    Q = w_p/tf.tile(tf.expand_dims(tf.reduce_sum(w_p, axis=-1), -1),[1, 1, n_gaussians])
    Q_per_d = tf.tile(tf.expand_dims(Q, -1), [1, 1, 1, D])

    # Compute derivatives and take max and min
    #Method 128: direct derivative formula (convertible to min-max)
    #s0 = tf.reduce_sum(Q, fv_noise)  # n_batches X n_gaussians
    #d_pi = (s0 - n_points * w_per_batch) / (tf.sqrt(w_per_batch) * n_points)
    d_pi_all = tf.expand_dims((Q - batch_w)/ (tf.sqrt(batch_w) * n_points), -1)
    d_pi = tf.concat(
        [tf.reduce_max(d_pi_all , axis=1), tf.reduce_sum(d_pi_all , axis=1)], axis=2)

    d_mu_all = Q_per_d * (batch_points - batch_mu) / batch_sig
    d_mu = (1 / (n_points * tf.sqrt(w_per_batch_per_d))) * tf.concat(
        [tf.reduce_max(d_mu_all , axis=1), tf.reduce_min(d_mu_all , axis=1), tf.reduce_sum(d_mu_all , axis=1)], axis=2)

    d_sig_all = Q_per_d * ( tf.pow((batch_points - batch_mu) / batch_sig,2) - 1)
    d_sigma = (1 / (n_points * tf.sqrt(2*w_per_batch_per_d))) * tf.concat(
        [tf.reduce_max(d_sig_all, axis=1), tf.reduce_min(d_sig_all, axis=1), tf.reduce_sum(d_mu_all , axis=1)], axis=2)

    #Power normaliation
    alpha = 0.5
    d_pi = tf.sign(d_pi) * tf.pow(tf.abs(d_pi),alpha)
    d_mu = tf.sign(d_mu) * tf.pow(tf.abs(d_mu), alpha)
    d_sigma = tf.sign(d_sigma) * tf.pow(tf.abs(d_sigma), alpha)

    # L2 normaliation
    d_pi = tf.nn.l2_normalize(d_pi, dim=1)
    d_mu = tf.nn.l2_normalize(d_mu, dim=1)
    d_sigma = tf.nn.l2_normalize(d_sigma, dim=1)

    if flatten:
        #flatten d_mu and d_sigma
        d_pi = tf.contrib.layers.flatten(tf.transpose(d_pi, perm=[0, 2, 1]))
        d_mu = tf.contrib.layers.flatten(tf.transpose(d_mu,perm=[0,2,1]))
        d_sigma = tf.contrib.layers.flatten(tf.transpose(d_sigma,perm=[0,2,1]))
        fv  = tf.concat([d_pi, d_mu, d_sigma], axis=1)
    else:
        fv = tf.concat([d_pi, d_mu, d_sigma], axis=2)
        fv = tf.transpose(fv, perm=[0, 2, 1])

    # fv = fv / tf.norm(fv)
    return fv


def fv_layer(input, n_gaussians, initialize='random', flatten=False, scope='fv'):
    D = input.shape[2].value
    w, mu, sigma = get_gmm_vars(n_gaussians, D=D, initialize=initialize, scope=scope)
    gmm_loss = get_gmm_loss(tf.concat(input,axis=0), w, mu, sigma, cp=0.8, cmu=0.1, csig=0.1, cw=0.1, scope = scope)
    tf.summary.scalar(scope+'gmm loss', gmm_loss)
    tf.add_to_collection('gmm_loss', gmm_loss)
    fv = get_fv_minmax(input, w, tf.transpose(mu), tf.transpose(sigma), flatten=flatten)
    return fv, w, mu, sigma