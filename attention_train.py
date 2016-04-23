"""Train attention model."""


import attention_model as model
from keras_utils import center, crps, real_to_cdf, vec_to_cdf, preprocess, rotation_augmentation, shift_augmentation

import random
import theano
import numpy as np
import sys

theano.config.floatX = 'float32'

X_TRAIN_FILE = 'keras_train_x.npy'
Y_TRAIN_FILE = 'keras_train_y.npy'
X_TEST_FILE = 'keras_test_x.npy'
Y_TEST_FILE = 'keras_test_y.npy'

NUM_FRAMES = 3
PREPROCESS = False
ROTATE = False
TEST_PROP = 0.15
ROTATION_ANGLE = 20
CROP_SIZE = 64
SEED = 888

def load_data(x_file, y_file):
    """
    Load training data from .npy files.
    """
    X = np.load(x_file).astype(theano.config.floatX)
    y = np.load(y_file).astype(theano.config.floatX)

    seed = SEED
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    new_y = np.zeros((len(y), 600, 2))
    new_y[:, :, 0] = y[:, 0, np.newaxis] <= np.arange(600).reshape((1, 600))
    new_y[:, :, 1] = y[:, 1, np.newaxis] <= np.arange(600).reshape((1, 600))

    return X.astype(theano.config.floatX), new_y.astype(theano.config.floatX)


def run():
    cdf_fn = vec_to_cdf
    lab_cdf_fn = lambda x: x

    print 'loading data'
    X_train, y_train = load_data(X_TRAIN_FILE, Y_TRAIN_FILE)
    X_test, y_test = load_data(X_TEST_FILE, Y_TEST_FILE)
    print 'train x', X_train.shape
    print 'train y', y_train.shape
    print 'test x', X_test.shape
    print 'test y', y_test.shape

    X_train_eval = center(X_train, CROP_SIZE)
    X_test_eval = center(X_test, CROP_SIZE)

    print 'loading model'
    input_dim = CROP_SIZE
    input_channels = X_train.shape[1]
    m = model.Attention(input_dim=(input_dim, input_dim),
                        input_channels=input_channels,
                        down_sample=4,
                        output_dim=600,
                        nh=100, std=0.02,
                        num_filters=1, filter_dim=3, pool_dim=1,
                        n_steps=4)

    m.setup_params()
    train, predict = m.get_functions()

    nb_iter = 1000
    epochs_per_iter = 1
    batch_size = 32
    calc_crps = 10
    noise_scale = 0.01 * np.mean(np.abs(X_train))

    # remember min val. losses (best iterations)
    min_val_loss = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i, nb_iter))
        print('-'*50)

        if ROTATE:
            print('Augmenting images - rotations')
            X_train_aug = rotation_augmentation(X_train, ROTATION_ANGLE)
        else:
            X_train_aug = X_train
        print('Augmenting images - shifts')
        X_train_aug = (shift_augmentation(X_train_aug, CROP_SIZE)).astype(
            theano.config.floatX)
        # adding noise -- this is necessary if a lot of entries in the matrix are 0
        X_train_aug += noise_scale * np.random.randn(*X_train_aug.shape)

        print('Fitting model...')
        losses = []
        rewards = []
        for idx in xrange(len(X_train_aug)):
            _, loss, reward = \
                train(X_train_aug[idx:idx + 1, ...], y_train[idx, :, 0], 0.01)
            losses.append(loss)
            rewards.append(reward)

        print 'avg loss', np.mean(losses)
        print 'avg reward', np.mean(rewards)
        print 'base reward', m.base_reward.get_value()
        print 'l0', m.l0.get_value()

        rand_idx = 0
        out, locations = predict(X_train_eval[idx:idx + 1, ...])
        print 'idx', rand_idx
        print 'locations', locations

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            train_pred = []
            for idx in xrange(len(X_train_eval)):
                out, _ = predict(X_train_eval[idx:idx + 1, ...])
                train_pred.append(out)

            val_pred = []
            for idx in xrange(len(X_test_eval)):
                out, _ = predict(X_test_eval[idx:idx + 1, ...])
                val_pred.append(out)

            # CDF for train and test data
            cdf_train = lab_cdf_fn(y_train[..., 0])
            cdf_test = lab_cdf_fn(y_test[..., 0])

            # CDF for predicted data
            cdf_pred = cdf_fn(train_pred)
            cdf_val_pred = cdf_fn(val_pred)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, cdf_pred)
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, cdf_val_pred)
            print('CRPS(test) = {0}'.format(crps_test))

            val_loss = crps_test


if __name__ == '__main__':
    run()
