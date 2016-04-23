#!/usr/bin/env python

"""Pipeline for generating data and training NN models."""

import create_lmdb
import nn_models
import solver

import caffe

import argparse
import os
import sys

TMP_DATA_DIR = '/mnt'
PROCESSED_DATA_DIR = '/mnt/processed_data'
PREPROC = create_lmdb.min_max_mean_preproc
TRANS = lambda a, b: [None]
MODEL_DIR = '/mnt/solvers'
MODEL = nn_models.lenet
BATCH_SIZE = 32
DEVICE_ID = 0


def main(model_name, label_key, slice_type, new_db):
    lmdb_train = os.path.join(TMP_DATA_DIR, 'lmdb_train_%s_%s' % (model_name, slice_type))
    lmdb_train_lab = os.path.join(TMP_DATA_DIR, 'lmdb_train_lab_%s_%s' % (model_name, slice_type))
    lmdb_test = os.path.join(TMP_DATA_DIR, 'lmdb_test_%s_%s' % (model_name, slice_type))
    lmdb_test_lab = os.path.join(TMP_DATA_DIR, 'lmdb_test_lab_%s_%s' % (model_name, slice_type))

    if new_db:
        train_data = os.path.join(PROCESSED_DATA_DIR, 'local_train')
        test_data = os.path.join(PROCESSED_DATA_DIR, 'local_test')

        print 'Creating train LMDB'
        create_lmdb.create_db(lmdb_train, lmdb_train_lab, train_data,
                              frames_preproc=PREPROC,
                              trans_generator=TRANS,
                              label_key=label_key,
                              slice_type=slice_type)

        print 'Creating test LMDB'
        create_lmdb.create_db(lmdb_test, lmdb_test_lab, test_data,
                              frames_preproc=PREPROC,
                              trans_generator=TRANS,
                              label_key=label_key,
                              slice_type=slice_type)

    print 'Creating neural net'
    full_model_name = '%s_%s_%s' % (model_name, label_key, slice_type)
    nn_models.write_model(full_model_name, MODEL_DIR, MODEL,
                          lmdb_train, lmdb_train_lab, lmdb_test, lmdb_test_lab,
                          BATCH_SIZE)

    print 'Training neural net'
    caffe.set_device(DEVICE_ID)
    caffe.set_mode_gpu()
    solver_path = os.path.join(MODEL_DIR, '%s_solver.prototxt' % full_model_name)
    solver.train(solver_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name')
    parser.add_argument('--label_key', default='systole', choices=['systole', 'diastole'])
    parser.add_argument('--slice_type', default='sax', choices=['sax', '2ch', '4ch'])
    parser.add_argument('--new_db', default=False, action='store_true')
    args = parser.parse_args()

    main(args.model_name, args.label_key, args.slice_type, args.new_db)
