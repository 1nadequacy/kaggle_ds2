"""Script for getting preprocessed files into numpy arrays for Keras."""

import common
import convert_data
import create_lmdb

import numpy as np
import os
import sys
import pickle
import random


def create_data(x_file, y_file, processed_data_folder,
                frames_preproc=None,
                slice_type='sax',
                one_slice_per_study=False,
                ids_file=None):
    """Creates numpy arrays for use by Keras.

    x_file: the destination file for the x numpy array.
    y_file: the destination file for the y (systole, diastole) numpy array.
    processed_data_folder: the directory of .pkl files written by data_pipeline.py.
    frames_preproc: any chosen preprocessing function for the frames.
    one_slice_per_study: only take one slice form each study
    ids_file: save ids to file

    """
    frames_preproc = frames_preproc or create_lmdb.default_preproc
    X = []
    y = []
    ids = []

    for study_file in os.listdir(processed_data_folder):
        assert study_file.endswith('.pkl'), 'file %s has wrong extension' % study_file
        with open(os.path.join(processed_data_folder, study_file), 'rb') as f:
            study = pickle.load(f)
            study_id = study['study']
            if slice_type not in study:
                print >>sys.stderr, 'study_id %s does not have %s' % (study_id, slice_type)
                continue
            slices = study[slice_type].keys()
            if one_slice_per_study:
                slices = slices[:1]
            for slice_ in slices:
                systole, diastole = study['systole'], study['diastole']
                frames = common.slice_to_numpy(study[slice_type][slice_])
                frames = frames_preproc(frames)
                X.append(frames)
                y.append([systole, diastole])
                ids.append(study_id)
                if len(X) % 100 == 0:
                    print 'added', len(X), 'data points to db'

    X = np.array(X)
    y = np.array(y)
    ids = np.array(ids)
    np.save(x_file, X)
    np.save(y_file, y)
    if ids_file:
        np.save(ids_file, ids)


def create_full_data(x_file, diffs_file, y_file,
                     processed_data_folder,
                     frames_preproc=None,
                     slice_type='sax',
                     ids_file=None):
    """Creates numpy arrays for use by Keras.

    x_file: the destination file for the x numpy array.
    y_file: the destination file for the y (systole, diastole) numpy array.
    processed_data_folder: the directory of .pkl files written by data_pipeline.py.
    frames_preproc: any chosen preprocessing function for the frames.
    ids_file: save ids to file

    """
    frames_preproc = frames_preproc or create_lmdb.default_preproc
    X = []
    diffs = []
    y = []
    ids = []

    for study_file in os.listdir(processed_data_folder):
        assert study_file.endswith('.pkl'), 'file %s has wrong extension' % study_file
        with open(os.path.join(processed_data_folder, study_file), 'rb') as f:
            study = pickle.load(f)
            study_id = study['study']
            if slice_type not in study:
                print >>sys.stderr, 'study_id %s does not have %s' % (study_id, slice_type)
                continue
            systole, diastole = study['systole'], study['diastole']
            slices = common.study_to_numpy(study, slice_type=slice_type)
            slice_locations = common.slice_locations(study, slice_type)

            my_frames = []
            for i in xrange(len(slices)):
                my_frames.append(frames_preproc(slices[i]))

            my_diffs = [0.0] * len(slice_locations)
            for i in xrange(len(slice_locations)):
                if i + 1 < len(slice_locations):
                    my_diffs[i] += 0.5 * (slice_locations[i + 1] - slice_locations[i])
                if i > 0:
                    my_diffs[i] += 0.5 * (slice_locations[i] - slice_locations[i - 1])

            X.append(np.array(my_frames))
            diffs.append(np.array(my_diffs))
            y.append(np.array([systole, diastole]))
            ids.append(study_id)
            if len(X) % 100 == 0:
                print 'added', len(X), 'data points to db'

    X = np.array(X)
    diffs = np.array(diffs)
    y = np.array(y)
    ids = np.array(ids)
    np.save(x_file, X)
    np.save(diffs_file, diffs)
    np.save(y_file, y)
    if ids_file:
        np.save(ids_file, ids)
