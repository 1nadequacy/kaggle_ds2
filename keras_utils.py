from __future__ import print_function

import validate

import numpy as np
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle
from scipy import ndimage
from keras.utils.generic_utils import Progbar


def crps(true, pred):
    """
    Calculation of CRPS.

    :param true: true values (labels)
    :param pred: predicted values
    """
    return np.sum(np.square(true - pred)) / true.size


def center(X, crop_size):
    if len(X.shape) != 4:
        new_X = []
        for i in xrange(len(X)):
            new_X.append(center(X[i], crop_size))
        return np.array(new_X)

    _, _, shape_x, shape_y = X.shape
    ctr_x, ctr_y = shape_x // 2, shape_y // 2
    return X[:, :, 
             ctr_x - crop_size // 2:ctr_x + (crop_size + 1) // 2,
             ctr_y - crop_size // 2:ctr_y + (crop_size + 1) // 2]


def real_to_cdf(y, sigma=1e-10):
    """
    Utility function for creating CDF from real number and sigma (uncertainty measure).

    :param y: array of real values
    :param sigma: uncertainty measure. The higher sigma, the more imprecise the prediction is, and vice versa.
    Default value for sigma is 1e-10 to produce step function if needed.
    """
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf


def vec_to_cdf(v, loss=0):
    for i in xrange(len(v)):
        v[i] = validate.submission_helper(v[i])
    return v


def preprocess(X):
    """
    Pre-process images that are fed to neural network.

    :param X: X
    """
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = denoise_tv_chambolle(X[i, j], weight=0.1, multichannel=False)
        progbar.add(1)
    return X


def rotation_augmentation(X, angle_range):
    X_rot = np.copy(X)
    if len(X.shape) != 4:
        for i in xrange(len(X)):
            X_rot[i] = rotation_augmentation(X[i], angle_range)
        return X_rot

    for i in range(len(X)):
        angle = np.random.randint(-angle_range, angle_range)
        for j in range(X.shape[1]):
            X_rot[i, j] = ndimage.rotate(X[i, j], angle, reshape=False, order=1)
    return X_rot


def shift_augmentation(X, crop_size):
    if len(X.shape) != 4:
        X_shift = []
        for i in xrange(len(X)):
            X_shift.append(shift_augmentation(X[i], crop_size))
        return np.array(X_shift)

    ndata, num_frames, shape_x, shape_y = X.shape
    X_shift = np.zeros((ndata, num_frames, crop_size, crop_size))
    size = X.shape[2:]
    for i in range(len(X)):
        offset_x = int(np.random.rand() * (shape_x - crop_size + 1))
        offset_y = int(np.random.rand() * (shape_y - crop_size + 1))
        for j in range(X.shape[1]):
            X_shift[i, j] = X[i, j, 
                              offset_x:offset_x + crop_size,
                              offset_y:offset_y + crop_size]
    return X_shift
