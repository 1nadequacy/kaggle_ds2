#!/usr/bin/env python

"""Pipeline for generating data and training NN models for many slices."""

import create_lmdb
import nn_models
import solver
import validate

import caffe

import argparse
import os
import sys
import subprocess


TMP_DATA_DIR = '/mnt2/lmdb'
PROCESSED_DATA_DIR = '/mnt2/full_converted'
PREPROC = create_lmdb.min_max_mean_preproc
TRANS = lambda a, b: [None]
MODEL_DIR = '/mnt2/models'
MODEL = nn_models.lenet_weight
BATCH_SIZE = 32
DEVICE_ID = 0
# bucketize all locations in 20 buckets and take those with more than 200 slices
SAX_LOCATION_BUCKETS = [-89.95, -72.17, -54.39, -36.61, -18.83, -1.05, 16.73, 34.50, 52.28, 70.06, 87.84, 105.62, 123.40]
CH_LOCATION_BUCKETS = [-1000000, 100000]


def keep_slice_location(bucket_id, buckets):
    assert bucket_id >= 0 and bucket_id < len(buckets), 'incorrect bucket id %s' % bucket_id
    left, right = buckets[bucket_id], buckets[bucket_id + 1]
    def func(slice_data):
        loc = slice_data[0]['slice_location']
        return loc >= left and loc <= right
    return func


def get_buckets(slice_type):
    return SAX_LOCATION_BUCKETS if slice_type == 'sax' else CH_LOCATION_BUCKETS


def create_dbs(model_name, label_key, slice_type, new_db):
    train_data = os.path.join(PROCESSED_DATA_DIR, 'local_train')
    test_data = os.path.join(PROCESSED_DATA_DIR, 'local_test')

    base_path = os.path.join(TMP_DATA_DIR, model_name, label_key, slice_type)
    subprocess.check_output(['mkdir', '-p', base_path])
    dbs = []
    buckets = get_buckets(slice_type)
    for bucket_id in range(len(buckets) - 1):
        lmdb_train = os.path.join(base_path, 'bucket_%s' % bucket_id, 'train')
        subprocess.check_output(['mkdir', '-p', lmdb_train])
        lmdb_test = os.path.join(base_path, 'bucket_%s' % bucket_id, 'test')
        subprocess.check_output(['mkdir', '-p', lmdb_test])
        dbs.append((lmdb_train, lmdb_test))

        keep_slice = keep_slice_location(bucket_id, buckets)

        if new_db:
            print 'Creating LMDBs for bucket %s' % bucket_id
            # creating train lmdb
            create_lmdb.create_db_unilabel(lmdb_train, train_data,
                                    frames_preproc=PREPROC,
                                    trans_generator=TRANS,
                                    label_key=label_key,
                                    slice_type=slice_type,
                                    keep_slice=keep_slice)

            # creating test lmdb
            create_lmdb.create_db_unilabel(lmdb_test, test_data,
                                    frames_preproc=PREPROC,
                                    trans_generator=TRANS,
                                    label_key=label_key,
                                    slice_type=slice_type,
                                    keep_slice=keep_slice)
    return dbs


def get_nns(model_name, label_key, slice_type, num_buckets, snapshot_id):
    nns = []
    base_path = os.path.join(MODEL_DIR, model_name, label_key, slice_type)
    for bucket_id in range(num_buckets):
        model_path = os.path.join(base_path, 'bucket_%s' % bucket_id)
        deploy_file = os.path.join(model_path, 'model_deploy.prototxt')
        model_file = os.path.join(model_path, 'model_iter_%s.caffemodel' % snapshot_id)
        nn = validate.get_net(deploy_file, model_file)
        nns.append(nn)
    return nns


def train_models(model_name, label_key, slice_type, dbs, niter):
    caffe.set_device(DEVICE_ID)
    caffe.set_mode_gpu()
    base_path = os.path.join(MODEL_DIR, model_name, label_key, slice_type)
    subprocess.check_output(['mkdir', '-p', base_path])

    for bucket_id in range(len(dbs)):
        print 'Creating model for bucket %s' % bucket_id
        lmdb_train, lmdb_test = dbs[bucket_id]
        model_path = os.path.join(base_path, 'bucket_%s' % bucket_id)
        subprocess.check_output(['mkdir', '-p', model_path])
        nn_models.write_model('model', model_path, MODEL,
                              lmdb_train, None, lmdb_test, None,
                              BATCH_SIZE)

        print 'Training model for bucket %s' % bucket_id
        solver_path = os.path.join(model_path, 'model_solver.prototxt')
        stats = solver.train(solver_path, niter=niter)
        with open(os.path.join(model_path, 'stats'), 'w') as f:
            for it in sorted(stats.keys()):
                print >>f, 'iter %s, crps %s, loss %s, smooth: %s' % (it, stats[it]['crps'], stats[it]['loss'], stats[it]['smooth'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--label_key', default='systole', choices=['systole', 'diastole'])
    parser.add_argument('--slice_type', default='sax', choices=['sax', '2ch', '4ch'])
    parser.add_argument('--new_db', default=False, action='store_true')
    parser.add_argument('--num_iters', default=30)
    args = parser.parse_args()

    dbs = create_dbs(args.model_name, args.label_key, args.slice_type, args.new_db)
    train_models(args.model_name, args.label_key, args.slice_type, dbs, int(args.num_iters))
