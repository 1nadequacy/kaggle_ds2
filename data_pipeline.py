#!/usr/bin/env python

"""Main pipeline for processing data."""

import fourier
import lv_segment
import common
import convert_data

from joblib import Parallel, delayed
import numpy as np
import subprocess
import sys
import os
import pickle
import random
from skimage import transform
import caffe
import time
import argparse

SEED = 888
TEST_PROP = 0.15
DEVICE_ID = 0

SCALE = (1.5, 1.5)  # desired scale of post-processed image
DIMENSION = (72, 72)  # desired dimensions of post-processed image
PAD = 0  # desired pad beyond ROI radius
CENTER_SLICE = False  # only use center slice


def process_rois(study,
                 slice_type='sax',
                 calc_rois=fourier.calc_rois,
                 radius=fourier.HEART_RADIUS,
                 pad=PAD):
    """Given study, performs ROI analysis and crops image to ROI"""

    rois, centers = calc_rois(study, slice_type, radius)
    for slice_, (ctr_x, ctr_y) in centers.items():
        frames = study[slice_type][slice_]
        for frame in frames:
            scale_x, scale_y = frame['scale_x'], frame['scale_y']
            roi_radius = int((radius + pad) / scale_x + 1)
            frame['pixel'] = np.pad(frame['pixel'],
                                    [(roi_radius, roi_radius),
                                     (roi_radius, roi_radius)],
                                    mode='edge')
            frame['pixel'] = frame['pixel'][ctr_x:ctr_x + 2 * roi_radius,
                                            ctr_y:ctr_y + 2 * roi_radius]


def process_sax(study_id, study, scale=SCALE, dim=DIMENSION,
                  use_lv_segmentation=False):
    """Preliminary image processing for study sax.

    scales images to desired scale (pixel spacing in mm).
    resizes images to desired dim (in pixels).
    use_lv_segmentation=True to use lv segmentation for finding ROIs.

    """
    if 'sax' not in study or not study['sax']:
        print >>sys.stderr, 'empty sax for study %s' % (study_id)
        return

    if CENTER_SLICE:
        cs = common.center_slices(study, 'sax')
        for slice_ in study['sax'].keys():
            if slice_ not in cs:
                del study['sax'][slice_]

    module = lv_segment if use_lv_segmentation else fourier
    process_rois(study, slice_type='sax', calc_rois=module.calc_rois,
        radius=module.HEART_RADIUS)
    for slice_ in study['sax']:
        frames = study['sax'][slice_]
        for frame in frames:
            frame['pixel'] = transform.resize(frame['pixel'], dim)


def process_ch(study_id, study, ch_type='2ch', scale=SCALE, dim=DIMENSION):
    """Preliminary image processing for study ch.

    scales images to desired scale (pixel spacing in mm).
    resizes images to desired dim (in pixels).

    """

    if ch_type not in study or not study[ch_type]:
        print >>sys.stderr, 'empty %s for study %s' % (ch_type, study_id)
        return

    process_rois(study, slice_type=ch_type, calc_rois=fourier.calc_rois, radius=fourier.HEART_RADIUS)
    for slice_ in study[ch_type]:
        frames = study[ch_type][slice_]
        for frame in frames:
            frame['pixel'] = transform.resize(frame['pixel'], dim)


def check_frames(study_id, study, slice_type):
    """Check if a slice has 30 frames."""
    if slice_type not in study:
        return
    for slice_ in study[slice_type].keys():
        if len(study[slice_type][slice_]) != 30:
            del study[slice_type][slice_]
            print >>sys.stderr, '%s slice %d in study %d does not have 30 time frames' % (slice_type, slice_, study_id)


def process_and_save_study(dest_folder,
                           study_id, study_path, systole, diastole,
                           use_lv_segmentation):
    metadata, slices = convert_data.get_slices(study_id, study_path)
    study = {'study': study_id,
             'systole': systole,
             'diastole': diastole}
    study.update(metadata)
    study.update(slices)

    for slice_type in convert_data.FOLDER_PATTERN.keys():
        check_frames(study_id, study, slice_type)

    process_sax(study_id, study, use_lv_segmentation=use_lv_segmentation)
    process_ch(study_id, study, ch_type='2ch')
    process_ch(study_id, study, ch_type='4ch')

    with open(os.path.join(dest_folder, 'study%d.pkl' % study_id), 'w') as f:
        pickle.dump(study, f)


def _process(dest_folder, study_id, study_path, systole, diastole, use_lv_segmentation):
    print 'processing study', study_id
    start = time.time()
    if systole is not None and study_id not in systole:
        print >>sys.stderr, 'no systole label for study', study_id
        return
    if diastole is not None and study_id not in diastole:
        print >>sys.stderr, 'no diastole label for study', study_id
        return

    lab_sys = systole[study_id] if systole else -1
    lab_dia = diastole[study_id] if diastole else -1
    process_and_save_study(dest_folder, study_id, study_path,
                           lab_sys, lab_dia, use_lv_segmentation)

    print 'saved study %s to %s, took %s s' % (study_id, dest_folder, time.time() - start)


def process_all(train_dest_folder, test_dest_folder, data_folder,
                labels_file, n_jobs=-1, use_lv_segmentation=False):
    random.seed(SEED)
    caffe.set_device(DEVICE_ID)
    caffe.set_mode_gpu()
    if labels_file:
        systole, diastole = convert_data.get_labels(labels_file)
    else:
        systole = diastole = None
    studies = sorted(convert_data.get_studies(data_folder))
    test_prop = TEST_PROP if test_dest_folder else 0.0
    dest_folders = ([test_dest_folder] * int(test_prop * len(studies)) +
                    [train_dest_folder] * (len(studies) - int(test_prop * len(studies))))
    random.shuffle(dest_folders)

    Parallel(n_jobs=n_jobs)(
        delayed(_process)(dest_folder, study_id, study_path, systole, diastole, use_lv_segmentation)
        for dest_folder, (study_id, study_path) in zip(dest_folders, studies))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dest_folder')
    parser.add_argument('data_folder')
    parser.add_argument('--labels_file', default=None, action='store')
    parser.add_argument('--n_jobs', default=-1,
        help='Number of threads to use', action='store')
    parser.add_argument('--lv_seg', default=False,
        help='To use lv segmentation', action='store_true')
    args = parser.parse_args()

    if args.labels_file is not None:
        train_local_dir = os.path.join(args.dest_folder, 'local_train')
        test_local_dir = os.path.join(args.dest_folder, 'local_test')
        subprocess.check_output(['mkdir', '-p', train_local_dir])
        subprocess.check_output(['mkdir', '-p', test_local_dir])
    else:
        train_local_dir = args.dest_folder
        test_local_dir = None
        subprocess.check_output(['mkdir', '-p', train_local_dir])

    process_all(train_local_dir, test_local_dir,
        args.data_folder, args.labels_file, int(args.n_jobs),
        use_lv_segmentation=args.lv_seg)
