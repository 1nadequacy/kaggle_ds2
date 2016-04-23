#!/usr/bin/env python

"""Pipeline for generating data and training NN models for many slices."""

import create_lmdb
import nn_models
import solver
import validate
import extract_features
import nn_pipeline_multi
import common

import caffe

from sklearn import ensemble, grid_search
import argparse
import os
import sys
import subprocess
import pickle
import numpy as np


BASE_DIR = '/mnt2/ensemble'
PROCESSED_DATA_DIR = '/mnt2/full_converted'
DEVICE_ID = 0
SIGMA = 22


def create_nets(model_name, label_key, snapshot_id=20000):
    nets = {}
    for slice_type in ['sax', '2ch', '4ch']:
        buckets = nn_pipeline_multi.get_buckets(slice_type)
        nns = nn_pipeline_multi.get_nns(model_name, label_key, slice_type, len(buckets) - 1, snapshot_id)
        nets[slice_type] = nns
    return nets


def scorer(estimator, X, y):
    y_hat = estimator.predict(X)
    assert y_hat.shape[0] == y.shape[0]
    result = 0.0
    for i in range(y.shape[0]):
        cdf = common.smooth_cdf(y_hat[i], SIGMA)
        result += common.crps_sample(y[i], cdf)
    return result / y.shape[0]


def get_fimps(estimator, names):
    fimps = estimator.feature_importances_
    assert fimps.shape[0] == names.shape[0]
    return sorted(zip(names, fimps), key=lambda x: -x[1])


def save(file_name, data):
    with open(file_name, 'w') as f:
        pickle.dump(data, f)


def load(file_name):
    with open(file_name, 'r') as f:
        return pickle.load(f)


def transform_data(model_name, data_name, data_folder, label_key='systole', new_db=False):
    base_path = os.path.join(BASE_DIR, model_name, label_key)
    subprocess.check_output(['mkdir', '-p', base_path])
    output_file = os.path.join(base_path, '%s.pkl' % data_name)
    if new_db:
        nets = {
            'systole': create_nets(model_name, 'systole'),
            'diastole': create_nets(model_name, 'diastole'),
        }

        print 'transforming data from %s' % data_folder

        y, x, names = extract_features.extract_features(data_folder, nets, model_name, label_key)
        assert y.shape[0] == x.shape[0]
        assert names.shape[0] == x.shape[1]
        save(output_file, [y, x, names])
    else:
        print 'loading data from %s' % output_file
        y, x, names = load(output_file)

    return y, x, names


def train(x, y, names, model_name, label_key, n_jobs):
    params = {
        'loss': ('ls',),
        'max_depth': (2, 3),
        'subsample': (0.5, 0.6, 0.7),
        'n_estimators' : (10, 20, 25, 30, 35, 40),
        'max_features': ('auto', 'sqrt', 'log2'),
        'learning_rate': (0.1, 0.2, 0.15, 0.05, 0.07)
    }
    gbm = ensemble.GradientBoostingRegressor()
    fitter = grid_search.GridSearchCV(
        gbm,
        params,
        verbose=1,
        n_jobs=n_jobs)
    fitter.fit(train_x, train_y)

    print 'best score: %s' % fitter.best_score_
    print 'best params: %s' % fitter.best_params_
    best_estimator = fitter.best_estimator_
    print 'fimps:'
    fimps = get_fimps(best_estimator, names)
    for name, val in fimps:
        print '%s: %s' % (name, val)

    base_path = os.path.join(BASE_DIR, model_name, label_key)
    subprocess.check_output(['mkdir', '-p', base_path])
    model_file = os.path.join(base_path, 'model.pkl')
    save(model_file, best_estimator)

    return best_estimator


def predict(model_file, data):
    estimator = load(model_file)
    pred = estimator.predict(data)
    cdfs = []
    for i in range(pred.shape[0]):
        cdfs.append(common.smooth_cdf(pred[i], SIGMA))
    return np.array(cdfs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--label_key', default='systole', choices=['systole', 'diastole'])
    parser.add_argument('--new_db', default=False, action='store_true')
    parser.add_argument('--n_jobs', default=4)
    args = parser.parse_args()

    caffe.set_device(DEVICE_ID)
    caffe.set_mode_gpu()

    train_data = os.path.join(PROCESSED_DATA_DIR, 'local_ens')
    train_y, train_x, names = transform_data(args.model_name, 'train',
        train_data, args.label_key, args.new_db)

    test_data = os.path.join(PROCESSED_DATA_DIR, 'local_test')
    test_y, test_x, names = transform_data(args.model_name, 'test',
        test_data, args.label_key, args.new_db)

    estimator = train(train_x, train_y, names, args.model_name, args.label_key, n_jobs=int(args.n_jobs))
    print 'train score: %s' % scorer(estimator, train_x, train_y)
    print 'test score: %s' % scorer(estimator, test_x, test_y)
