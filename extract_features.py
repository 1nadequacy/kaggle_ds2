"""Feature extraction for final ensemble.

For each model, add a method that computes that feature and use that method
to extract_features_for_study.
"""

import common
import convert_data
import validate
import nn_pipeline_multi
import create_lmdb

import numpy as np
import os


def extract_features_for_study(study, nets, model_name):
    extractors = [meta_features,
                  slice_meta_features,
                  lambda study: nn_features(study, nets['systole'], model_name + '_sys', 'sax', create_lmdb.min_max_mean_preproc),
                  lambda study: nn_features(study, nets['diastole'], model_name + '_dia', 'sax', create_lmdb.min_max_mean_preproc),
                  lambda study: nn_features(study, nets['systole'], model_name + '_sys', '2ch', create_lmdb.min_max_mean_preproc),
                  lambda study: nn_features(study, nets['systole'], model_name + '_sys', '4ch', create_lmdb.min_max_mean_preproc),
                  lambda study: nn_features(study, nets['diastole'], model_name + '_dia', '2ch', create_lmdb.min_max_mean_preproc),
                  lambda study: nn_features(study, nets['diastole'], model_name + '_dia', '4ch', create_lmdb.min_max_mean_preproc),
                  # add your extractors here
                  ]

    features, names = [], []
    for extractor in extractors:
        fe, na = extractor(study)
        features.append(fe)
        names.append(na)
    return np.concatenate(features, axis=0), np.concatenate(names, axis=0)


def extract_features(processed_data_folder, nns, model_name, label_key='systole'):
    data = []
    for study_file in os.listdir(processed_data_folder):
        print 'processing %s' % (study_file)
        assert study_file.endswith('.pkl'), 'file %s has wrong extension' % study_file
        study = common.load_study_from_pickle(os.path.join(processed_data_folder, study_file))
        label = study.get(label_key, -1)
        features, names = extract_features_for_study(study, nns, model_name)
        data.append((study['study'], label, features))
    data = sorted(data)
    labels, features = [], []
    for _, l, f in data:
        labels.append(l)
        features.append(f)
    return np.array(labels), np.array(features), names


def min_max_mean(ll):
    return np.min(ll), np.max(ll), np.mean(ll)


def meta_features(study):
    names = np.array(['age', 'sex'])

    if 'patient_age' in study:
        age = study['patient_age']
    else:
        age = -1

    if 'patient_sex' in study:
        assert study['patient_sex'] in ['M', 'F']
        sex = 0 if study['patient_sex'] == 'M' else 1
    else:
        sex = -1
    return np.array([age, sex]), names


def slice_meta_features(study, slice_type='sax'):
    names = np.array([
        'num_slices',
        'max_loc_diff',
        'num_unique_loc',
        'min_diff',
        'max_diff',
        'mean_diff',
        'slice_thickness',
        'scale_x',
        'scale_y',
        'shape_x',
        'shape_y',
    ])
    if slice_type not in study:
        return np.array([0] * 11), names
    num_slices = len(study[slice_type])

    # slice locations
    slice_locations = common.slice_locations(study, slice_type)
    max_location_difference = max(slice_locations) - min(slice_locations)
    unique_slice_locations = sorted(set(int(s + 0.1) for s in slice_locations))
    num_unique_locations = len(unique_slice_locations)
    if num_slices > 1:
        min_diff, max_diff, mean_diff = min_max_mean(
                [unique_slice_locations[i + 1] - unique_slice_locations[i]
                 for i in xrange(len(unique_slice_locations) - 1)])
    else:
        min_diff, max_diff, mean_diff = 0, 0, 0

    # frame metadata
    one_frame = study[slice_type].values()[0][0]
    slice_thickness = one_frame['slice_thickness']
    scale_x = one_frame['scale_x']
    scale_y = one_frame['scale_y']
    shape_x, shape_y = one_frame['pixel'].shape

    return np.array([num_slices,
                     max_location_difference, num_unique_locations,
                     min_diff, max_diff, mean_diff,
                     slice_thickness, scale_x, scale_y,
                     shape_x, shape_y]), names


def nn_features(study, nets, model_name, slice_type, frames_preproc):
    nns = nets[slice_type]
    names = np.array(['%s_%s_bucket_%s' % (model_name, slice_type, i) for i in range(len(nns))])
    buckets = nn_pipeline_multi.get_buckets(slice_type)
    assert len(buckets) - 1 == len(nns), 'number of buckets and nns does not match'
    features = np.zeros(len(nns))
    if slice_type not in study:
        return features, names
    for bucket_id in range(len(nns)):
        keep_slice = nn_pipeline_multi.keep_slice_location(bucket_id, buckets)
        slices = study[slice_type].keys()
        pred, num = 0.0, 0
        for slice_id in slices:
            if keep_slice(study[slice_type][slice_id]):
                frames = common.slice_to_numpy(study[slice_type][slice_id])
                frames = frames_preproc(frames)
                pred += validate.net_output(nns[bucket_id], frames)
                num += 1
        if num > 0:
            pred /= num
        features[bucket_id] = pred
    return features, names
