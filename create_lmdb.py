"""Script for getting preprocessed files into Caffe-ready lmdb."""

import common
import convert_data
import pca

import caffe
import lmdb
import numpy as np
import os
import sys
import pickle
import random
import shutil
from scipy.fftpack import fftn, ifftn
from skimage import transform


def create_db(db_folder, db_label_folder, processed_data_folder,
              frames_preproc=None,
              trans_generator=lambda a, b: [None],
              label_key='systole',
              slice_type='sax',
              num_total_data=None,
              one_slice_per_study=False):
    """Creates LMDB for data and labels.

    db_folder: the LMDB destination for the data.
    db_label_folder: the LMDB destination for the labels.
    processed_data_folder: the directory of .pkl files written by data_pipeline.py.
    frames_preproc: any chosen preprocessing function for the frames.
    trans_generator: a function that takes in study and slice id and returns a set
        of transformations to include in the dataset.
    label_key: 'systole' or 'diastole'.
    num_total_data: total amount of data, if known.  Can't shuffle data without this.
    one_slice_per_study: only take one slice form each study

    """
    frames_preproc = frames_preproc or default_preproc

    for folder in [db_folder, db_label_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)

    data_db = lmdb.open(db_folder, map_size=1e12)
    label_db = lmdb.open(db_label_folder, map_size=1e12)

    if num_total_data:
        shuffled_counters = range(num_total_data)
        random.shuffle(shuffled_counters)

    # txn is a transaction object
    with data_db.begin(write=True) as data_txn:
        with label_db.begin(write=True) as label_txn:
            counter = 0
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
                        for trans in trans_generator(study_id, slice_):
                            volume = study[label_key]
                            volume, frames = apply_transform(
                                volume,
                                common.slice_to_numpy(study[slice_type][slice_]),
                                trans)
                            frames = frames_preproc(frames)
                            data = caffe.io.array_to_datum(frames)
                            label = np.array(volume <= np.arange(600),
                                             dtype=np.uint8).reshape((1, 1, 600))
                            label = caffe.io.array_to_datum(label)
                            if num_total_data:
                                str_id = '{:06d}'.format(shuffled_counters[counter])
                            else:
                                str_id = '{:06d}'.format(counter)
                            data_txn.put(str_id, data.SerializeToString())
                            label_txn.put(str_id, label.SerializeToString())
                            counter += 1
                            if counter % 100 == 0:
                                print 'added', counter, 'data points to db'

    label_db.close()
    data_db.close()


def create_db_unilabel(db_folder, processed_data_folder,
              frames_preproc=None,
              trans_generator=lambda a, b: [None],
              label_key='systole',
              slice_type='sax',
              keep_slice=lambda study: True):
    """Creates LMDB for data and labels, where labes is a single value.

    db_folder: the LMDB destination for the data.
    db_label_folder: the LMDB destination for the labels.
    processed_data_folder: the directory of .pkl files written by data_pipeline.py.
    frames_preproc: any chosen preprocessing function for the frames.
    trans_generator: a function that takes in study and slice id and returns a set
        of transformations to include in the dataset.
    label_key: 'systole' or 'diastole'.
    keep_slice: a function that decides if to keep a slice or not.

    """
    frames_preproc = frames_preproc or default_preproc

    if os.path.exists(db_folder):
        shutil.rmtree(db_folder)

    data_db = lmdb.open(db_folder, map_size=1e12)

    # txn is a transaction object
    with data_db.begin(write=True) as data_txn:
        counter = 0
        for study_file in os.listdir(processed_data_folder):
            assert study_file.endswith('.pkl'), 'file %s has wrong extension' % study_file
            with open(os.path.join(processed_data_folder, study_file), 'rb') as f:
                study = pickle.load(f)
                study_id = study['study']
                if slice_type not in study:
                    print >>sys.stderr, 'study_id %s does not have %s' % (study_id, slice_type)
                    continue
                slices = study[slice_type].keys()
                for slice_id in slices:
                    if not keep_slice(study[slice_type][slice_id]):
                        continue
                    for trans in trans_generator(study_id, slice_id):
                        volume = study[label_key]
                        volume, frames = apply_transform(
                            volume,
                            common.slice_to_numpy(study[slice_type][slice_id]),
                            trans)
                        frames = frames_preproc(frames)
                        label = int(volume)
                        data = caffe.io.array_to_datum(frames, label)
                        str_id = '{:06d}'.format(counter)
                        data_txn.put(str_id, data.SerializeToString())
                        counter += 1
        print '%s, added %s slices' % (db_folder, counter)

    data_db.close()


def apply_transform(volume, frames, trans):
    if not trans:
        return volume, frames

    num_frames, shape_x, shape_y = frames.shape

    if 'pca' in trans:
        noise = trans['pca']
        for i in xrange(num_frames):
            frames[i] = np.clip(frames[i] + noise, 0, 255)

    if 'scale' in trans:
        scale = trans['scale']
        new_ctr_x, new_ctr_y = int(scale * shape_x / 2), int(scale * shape_y / 2)
        for i in xrange(num_frames):
            if scale >= 1.0:
                frames[i] = transform.rescale(frames[i], scale) \
                    [new_ctr_x - shape_x // 2:new_ctr_x + (shape_x + 1) // 2,
                     new_ctr_y - shape_y // 2:new_ctr_y + (shape_y + 1) // 2]
            else:
                pad_x = shape_x - int(scale * shape_x) + 1
                pad_y = shape_y - int(scale * shape_y) + 1
                frames[i] = np.pad(transform.rescale(frames[i], scale),
                                   [[pad_x, pad_x], [pad_y, pad_y]],
                                   mode='edge') \
                    [pad_x + new_ctr_x - shape_x // 2:pad_x + new_ctr_x + (shape_x + 1) // 2,
                     pad_y + new_ctr_y - shape_y // 2:pad_y + new_ctr_y + (shape_y + 1) // 2]

        # scale volume proportionally
        volume = volume * scale ** 3

    if 'rotation' in trans:
        rotation = trans['rotation']
        for i in xrange(num_frames):
            frames[i] = transform.rotate(frames[i], rotation, mode='nearest')

    return volume, frames

def default_preproc(frames):
    return np.array(frames, dtype=np.uint8)

def time_delta_preproc(frames):
    new_frames = []
    for i in xrange(30):
        new_frames.append(frames[i].astype(float) / np.max(frames[i]))
    return np.array([new_frames[i + 1] - new_frames[i] for i in xrange(29)])

def min_max_mean_preproc(frames):
    return np.array([
        np.min(frames, axis=0), np.max(frames, axis=0), np.mean(frames, axis=0)],
        dtype='uint8')

def harmonic_preproc(frames):
    ff = fftn(frames)
    return np.array(
        [np.absolute(ifftn(ff[i])) for i in xrange(1, 4)],
        dtype=np.float)

def _rand(trans, i, study_id, slice_):
    return hash('{}{}{:03d}{:03d}'.format(i, trans, study_id, slice_))

def combined_trans_generator(*generators):

    def generator(study_id, slice_):
        for gens in zip(*[list(gen(study_id, slice_)) for gen in generators]):
            trans = dict(sum([gen.items() for gen in gens], []))
            yield trans

    return generator

def get_rotation_generator(k, min_rot, max_rot):
    # angles are in degrees

    def rot_generator(study_id, slice_):
        for i in xrange(k):
            rand = _rand('rotation', i, study_id, slice_)
            yield {'rotation': min_rot + (max_rot - min_rot) * (rand % 100) / 100.}

    return rot_generator

def get_scale_generator(k, min_scale, max_scale):

    def trans_generator(study_id, slice_):
        for i in xrange(k):
            rand = _rand('scale', i, study_id, slice_)
            yield {'scale': min_scale + (max_scale - min_scale) * (rand % 100) / 100.}

    return trans_generator

def get_pca_generator(k, V, eig, std_dev=0.1):

    def pca_generator(study_id, slice_):
        for i in xrange(k):
            rand = _rand('pca', i, study_id, slice_)  # TODO(make deterministic)
            yield {'pca': pca.pca_noise(V, eig, std_dev)}

    return pca_generator
