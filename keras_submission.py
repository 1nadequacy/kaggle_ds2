from __future__ import print_function

import convert_data
import validate

import csv
import numpy as np

from keras_model import get_model, get_lenet, get_leaky_model
from keras_utils import center, real_to_cdf, preprocess, vec_to_cdf

import theano
theano.config.floatX = 'float32'

OUT_SUBMISSION = 'submission7.csv'
SAMPLE_SUBMISSION = 'sample_submission_validate.csv'
LABELS_FILE = '/mnt/heartvol_readonly/train.csv'

GET_MODEL = get_model
OUT = 600

WEIGHTS_PREFIX = 'c_'
X_VAL_FILE = 'keras_val_x.npy'
IDS_FILE = 'keras_val_ids.npy'

NUM_FRAMES = 3
PREPROCESS = False
CROP_SIZE = 64

SAVE_BEST_WEIGHTS_SYS = '%sweights_systole_best.hdf5' % WEIGHTS_PREFIX
SAVE_BEST_WEIGHTS_DIA = '%sweights_diastole_best.hdf5' % WEIGHTS_PREFIX


def load_validation_data(x_file, ids_file):
    """
    Load validation data from .npy files.
    """
    X = np.load(x_file)
    ids = np.load(ids_file)

    X = X.astype(theano.config.floatX)

    return X, ids


def accumulate_study_results(ids, prob):
    """
    Accumulate results per study (because one study has many SAX slices),
    so the averaged CDF for all slices is returned.
    """
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    for i in range(size):
        study_id = ids[i]
        idx = int(study_id)
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]), dtype=np.float32)
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
        assert sum_result[i].shape[1] == 600
        assert np.all(np.diff(sum_result[i]) >= 0)
    return sum_result


def submission():
    """
    Generate submission file for the trained models.
    """
    cdf_fn = vec_to_cdf

    sys, dia = convert_data.get_labels(LABELS_FILE)
    all_systole = np.array(sys.values())
    all_diastole = np.array(dia.values())

    default_sys = validate.get_default(all_systole)
    default_dia = validate.get_default(all_diastole)

    print('Loading and compiling models...')
    model_systole = GET_MODEL(NUM_FRAMES, CROP_SIZE, CROP_SIZE, OUT)
    model_diastole = GET_MODEL(NUM_FRAMES, CROP_SIZE, CROP_SIZE, OUT)

    print('Loading models weights...')
    model_systole.load_weights(SAVE_BEST_WEIGHTS_SYS)
    model_diastole.load_weights(SAVE_BEST_WEIGHTS_DIA)

    print('Loading validation data...')
    X, ids = load_validation_data(X_VAL_FILE, IDS_FILE)

    if PREPROCESS:
        print('Pre-processing images...')
        X = preprocess(X)

    X = center(X, CROP_SIZE)

    batch_size = 32
    print('Predicting on validation data...')
    pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)
    pred_diastole = model_diastole.predict(X, batch_size=batch_size, verbose=1)
    print(pred_systole.shape)
    print(pred_diastole.shape)

    # real predictions to CDF
    cdf_pred_systole = cdf_fn(pred_systole)
    cdf_pred_diastole = cdf_fn(pred_diastole)
    print(cdf_pred_systole.shape)
    print(cdf_pred_diastole.shape)

    print('Accumulating results...')
    sub_systole = accumulate_study_results(ids, cdf_pred_systole)
    sub_diastole = accumulate_study_results(ids, cdf_pred_diastole)

    # write to submission file
    print('Writing submission to file...')
    fi = csv.reader(open(SAMPLE_SUBMISSION))
    f = open(OUT_SUBMISSION, 'w')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        key, target = idx.split('_')
        key = int(key)
        out = [idx]
        if key in sub_systole:
            if target == 'Diastole':
                out.extend(list(sub_diastole[key][0]))
            else:
                out.extend(list(sub_systole[key][0]))
        else:
            if target == 'Diastole':
                out.extend(list(default_dia))
            else:
                out.extend(list(default_sys))
            print('Miss {0}'.format(idx))
        fo.writerow(out)
    f.close()

    print('Done.')

if __name__ == '__main__':
    submission()
