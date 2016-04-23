"""For preparing submissions."""

import common
import create_lmdb

import caffe
import csv
import numpy as np
import os
import pickle
import sys


def get_default(all_labels):
    """Returns default prediction given distribution of volumes."""
    h = np.zeros(600)
    for j in np.ceil(all_labels).astype(int):
        h[j:] += 1
    h /= len(all_labels)
    return h


def get_net(deploy_proto_file, model_file):
    assert os.path.exists(deploy_proto_file), deploy_proto_file
    assert os.path.exists(model_file), model_file

    return caffe.Net(deploy_proto_file, model_file, caffe.TEST)


def net_output(net, frames, crop_size=64):
    _, shape_x, shape_y = frames.shape
    frames = np.expand_dims(frames, axis=0)
    net.blobs['data'].data[...] = \
        frames[:, :,
               shape_x // 2 - crop_size // 2: shape_x // 2 + (crop_size + 1) // 2,
               shape_y // 2 - crop_size // 2: shape_y // 2 + (crop_size + 1) // 2]
    net.forward()
    out = net.blobs['out'].data
    return out.reshape(out.size)


def submission_helper(pred, window=1):
    """Ensures CDF meets contest rules."""
    ret = []
    cdf = 0.0
    for i in xrange(600):
        cur_pred = np.mean(pred[i:i + window])
        cdf = max(cdf, min(1.0, cur_pred))
        ret.append(cdf)
    ret = [0] * (window // 2) + ret[:len(ret) - (window - 1) // 2]
    assert len(ret) == 600
    assert np.all(np.diff(ret) >= 0)
    return ret


def write_submission(out_file, systole_cdfs, diastole_cdfs):
    with open(out_file, 'w') as out:
        submission = csv.writer(out, lineterminator='\n')
        submission.writerow(['Id'] + ['P%d' % i for i in xrange(600)])
        for i in xrange(200):
            study_id = 501 + i
            for label_key, cdf in zip(['Diastole', 'Systole'],
                                      [diastole_cdfs[i], systole_cdfs[i]]):
                submission.writerow(['%d_%s' % (study_id, label_key)] +
                                    submission_helper(cdf))


def write_nn_submission(out_file, processed_data_folder,
                        sys_net, dia_net,
                        default_sys=None, default_dia=None,
                        frames_preproc=None):

    frames_preproc = frames_preproc or create_lmdb.default_preproc
    default_sys = np.zeros(600) if default_sys is None else default_sys
    default_dia = np.zeros(600) if default_dia is None else default_dia

    with open(out_file, 'w') as out:
        submission = csv.writer(out, lineterminator='\n')
        submission.writerow(['Id'] + ['P%d' % i for i in xrange(600)])
        for i in xrange(501, 701):
            print i
            study_file = os.path.join(processed_data_folder, 'study%d.pkl' % i)
            assert study_file.endswith('.pkl'), \
                'file %s has wrong extension' % study_file
            with open(os.path.join(processed_data_folder, study_file), 'rb') as f:
                study = pickle.load(f)
                study_id = study['study']
                assert study_id == i, i

                if 'sax' not in study:
                    print >>sys.stderr, 'no slices for study %d' % study_id
                    submission.writerow(['%d_Diastole' % study_id] +
                                         submission_helper(default_dia))
                    submission.writerow(['%d_Systole' % study_id] +
                                         submission_helper(default_sys))
                    continue

                study_data = study['sax']
                for label_key, net in zip(['Diastole', 'Systole'],
                                          [dia_net, sys_net]):
                    net_predictions = np.array(
                        [net_output(
                            net,
                            frames_preproc(common.slice_to_numpy(study_data[slice_])))
                         for slice_ in study_data])
                    final_predictions = (np.mean(net_predictions, axis=0)
                                         if len(net_predictions.shape) > 1
                                         else net_predictions)
                    submission.writerow(['%d_%s' % (study_id, label_key)] +
                                        submission_helper(final_predictions))
