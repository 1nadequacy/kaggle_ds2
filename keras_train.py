"""Usage: THEANO_FLAGS=device=gpu python keras_train.py

Install Theano with: sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
Install Keras with: sudo pip install git+git://github.com/fchollet/keras.git --upgrade --no-deps

"""

from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras_model import get_model, get_lenet, get_leaky_model, get_full
from keras_utils import center, crps, real_to_cdf, vec_to_cdf, preprocess, rotation_augmentation, shift_augmentation

import theano
theano.config.floatX = 'float32'

GET_MODEL = get_full

VALIDATION_OUT = 'val_loss.txt'
WEIGHTS_PREFIX = 'a_'

OUT = 1
USE_CRPS = True

X_TRAIN_FILE = 'keras_d_train_x.npy'
Y_TRAIN_FILE = 'keras_d_train_y.npy'
DIFFS_TRAIN_FILE = 'keras_d_train_d.npy'
X_TEST_FILE = 'keras_d_test_x.npy'
Y_TEST_FILE = 'keras_d_test_y.npy'
DIFFS_TEST_FILE = 'keras_d_test_d.npy'

NUM_FRAMES = 3
PREPROCESS = False
TEST_PROP = 0.15
ROTATION_ANGLE = 20
CROP_SIZE = 64
SEED = 888

SAVE_WEIGHTS_SYS = '%sweights_systole.hdf5' % WEIGHTS_PREFIX
SAVE_WEIGHTS_DIA = '%sweights_diastole.hdf5' % WEIGHTS_PREFIX
SAVE_BEST_WEIGHTS_SYS = '%sweights_systole_best.hdf5' % WEIGHTS_PREFIX
SAVE_BEST_WEIGHTS_DIA = '%sweights_diastole_best.hdf5' % WEIGHTS_PREFIX


def load_data(x_file, y_file, diffs_file=None):
    """
    Load training data from .npy files.
    """
    X = np.load(x_file)
    y = np.load(y_file)
    if diffs_file:
        diffs = np.load(diffs_file)

    seed = SEED
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    if diffs_file:
        np.random.seed(seed)
        np.random.shuffle(diffs)

    if OUT > 1:
        new_y = np.zeros((len(y), OUT, 2))
        new_y[:, :, 0] = y[:, 0, np.newaxis] <= np.arange(OUT).reshape((1, OUT))
        new_y[:, :, 1] = y[:, 1, np.newaxis] <= np.arange(OUT).reshape((1, OUT))
        y = new_y

    if diffs_file:
        return X, diffs, y

    return X, y


def split_data(X, y, split_ratio=0.15):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    X_train = X[split:, :, :, :]
    y_train = y[split:, :]

    return X_train, y_train, X_test, y_test


def train():
    """
    Training systole and diastole models.
    """
    cdf_fn = real_to_cdf if OUT == 1 else vec_to_cdf
    lab_cdf_fn = real_to_cdf if OUT == 1 else lambda x: x

    print('Loading training data...')
    X_train, diffs_train, y_train = load_data(X_TRAIN_FILE, Y_TRAIN_FILE, DIFFS_TRAIN_FILE)
    X_test, diffs_test, y_test = load_data(X_TEST_FILE, Y_TEST_FILE, DIFFS_TEST_FILE)

    if PREPROCESS:
        print('Pre-processing images...')
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)

    print('X_train shape', X_train.shape)
    print('y_train shape', y_train.shape)
    print('X_test shape', X_test.shape)
    print('y_test shape', y_test.shape)

    X_train_eval = center(X_train, CROP_SIZE)
    X_test_eval = center(X_test, CROP_SIZE)

    print('Loading and compiling models...')
    model_systole = GET_MODEL(NUM_FRAMES, CROP_SIZE, CROP_SIZE, OUT)
    model_diastole = GET_MODEL(NUM_FRAMES, CROP_SIZE, CROP_SIZE, OUT)

    nb_iter = 200
    epochs_per_iter = 1
    batch_size = 1 if DIFFS_TRAIN_FILE else 32
    calc_crps = 1  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        print('Augmenting images - rotations')
        X_train_aug = rotation_augmentation(X_train, ROTATION_ANGLE)
        print('Augmenting images - shifts')
        X_train_aug = shift_augmentation(X_train_aug, CROP_SIZE)

        print('Fitting systole model...')
        hist_systole = model_systole.fit(
                [X_train_aug, diffs_train], y_train[..., 0],
                shuffle=True, nb_epoch=epochs_per_iter, batch_size=batch_size,
                validation_data=([X_test_eval, diffs_test], y_test[..., 0]))

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit(
                [X_train_aug, diffs_train], y_train[..., 1],
                shuffle=True, nb_epoch=epochs_per_iter, batch_size=batch_size,
                validation_data=([X_test_eval, diffs_test], y_test[..., 1]))

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict(
                    [X_train_eval, diffs_train], batch_size=batch_size,
                    verbose=1)
            pred_diastole = model_diastole.predict(
                    [X_train_eval, diffs_train], batch_size=batch_size,
                    verbose=1)
            val_pred_systole = model_systole.predict(
                    [X_test_eval, diffs_test], batch_size=batch_size,
                    verbose=1)
            val_pred_diastole = model_diastole.predict(
                    [X_test_eval, diffs_test], batch_size=batch_size,
                    verbose=1)

            # CDF for train and test data (actually a step function)
            cdf_train_sys = lab_cdf_fn(y_train[..., 0])
            cdf_train_dia = lab_cdf_fn(y_train[..., 1])
            cdf_test_sys = lab_cdf_fn(y_test[..., 0])
            cdf_test_dia = lab_cdf_fn(y_test[..., 1])

            # CDF for predicted data
            cdf_pred_systole = cdf_fn(pred_systole, loss_systole ** 0.5)
            cdf_pred_diastole = cdf_fn(pred_diastole, loss_diastole ** 0.5)
            cdf_val_pred_systole = cdf_fn(val_pred_systole, val_loss_systole ** 0.5)
            cdf_val_pred_diastole = cdf_fn(val_pred_diastole, val_loss_diastole ** 0.5)

            # evaluate CRPS on training data
            crps_train_sys = crps(cdf_train_sys, cdf_pred_systole)
            crps_train_dia = crps(cdf_train_dia, cdf_pred_diastole)
            print('CRPS(train, sys) = {0}'.format(crps_train_sys))
            print('CRPS(train, dia) = {0}'.format(crps_train_dia))

            # evaluate CRPS on test data
            crps_test_sys = crps(cdf_test_sys, cdf_val_pred_systole)
            crps_test_dia = crps(cdf_test_dia, cdf_val_pred_diastole)
            print('CRPS(test, sys) = {0}'.format(crps_test_sys))
            print('CRPS(test, dia) = {0}'.format(crps_test_dia))

            if USE_CRPS:
                val_loss_systole = crps_test_sys
                val_loss_diastole = crps_test_dia

        print('Saving weights...')
        # save weights so they can be loaded later
        model_systole.save_weights(SAVE_WEIGHTS_SYS, overwrite=True)
        model_diastole.save_weights(SAVE_WEIGHTS_DIA, overwrite=True)

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            print('Saving BEST WEIGHTS systole')
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights(SAVE_BEST_WEIGHTS_SYS, overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            print('Saving BEST WEIGHTS diastole')
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights(SAVE_BEST_WEIGHTS_DIA, overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open(VALIDATION_OUT, mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))

    print('best systole:', min_val_loss_systole)
    print('best diastole:', min_val_loss_diastole)


if __name__ == '__main__':
    train()
