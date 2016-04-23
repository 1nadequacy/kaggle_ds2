#!/usr/bin/env python

"""Pipeline for generating submission."""

import convert_data
import create_lmdb
import validate

import numpy as np
import os
import sys

# NOTE(minmax): Assumes Fourier ROI preprocessing with
#               SCALE (1.5, 1.5), PAD 20, DIM (100, 100).
# NOTE(harmonic): Assumes Fourier ROI preprocessing with
#               SCALE (1.5, 1.5), PAD 10, DIM (74, 74).

SYS_DEPLOY_FILE = './minmax/minmax_deploy.prototxt'
SYS_MODEL_FILE = './minmax/minmax_iter_8000.caffemodel'

DIA_DEPLOY_FILE = './minmax/minmaxdia_deploy.prototxt'
DIA_MODEL_FILE = './minmax/minmaxdia_iter_8000.caffemodel'

PROCESSED_VALIDATION_DATA_DIR = '/mnt/processed_data/validate'
PREPROC = create_lmdb.min_max_mean_preproc

LABELS_FILE = '/mnt/heartvol_readonly/train.csv'

def main(out_file):
    sys, dia = convert_data.get_labels(LABELS_FILE)
    all_systole = np.array(sys.values())
    all_diastole = np.array(dia.values())

    default_sys = validate.get_default(all_systole)
    default_dia = validate.get_default(all_diastole)

    sys_net = validate.get_net(SYS_DEPLOY_FILE, SYS_MODEL_FILE)
    dia_net = validate.get_net(DIA_DEPLOY_FILE, DIA_MODEL_FILE)

    print 'Writing submission'
    validate.write_nn_submission(
        out_file, PROCESSED_VALIDATION_DATA_DIR,
        sys_net, dia_net,
        default_sys, default_dia,
        PREPROC)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'Usage: %s out_file' % sys.argv[0]
        sys.exit(1)

    main(sys.argv[1])
