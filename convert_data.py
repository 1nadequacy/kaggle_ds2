import os
import sys
import re
import pandas as pd
import numpy as np
try:
    import pydicom
except:
    import dicom as pydicom
import pickle
from collections import defaultdict


FRAME_NAME_PATTERN = 'IM-(\d{4,})-(\d{4})\.dcm'
FOLDER_PATTERN = {
    'sax': 'sax_(\d+)',
    '2ch': '2ch_(\d+)',
    '4ch': '4ch_(\d+)',
}


def get_labels(file_name):
    labels_csv = pd.read_csv(file_name)
    systole = {key : value for key, value in\
        zip(labels_csv['Id'].values, labels_csv['Systole'].values)}
    diastole = {key : value for key, value in\
        zip(labels_csv['Id'].values, labels_csv['Diastole'].values)}
    return systole, diastole


def get_studies(folder_name):
    return [(int(study), os.path.join(folder_name, study, 'study'))
            for study in os.listdir(folder_name)]


def pixels_from_dicom(f):
    window_min = int(f.WindowCenter) - 0.5 * int(f.WindowWidth)
    img = 255 * np.clip(
        (f.pixel_array.astype(np.float) - window_min) / float(f.WindowWidth),
        0.0, 1.0)
    return img


def get_frame(slice_path, frame_file, metadata):
    match = re.match(FRAME_NAME_PATTERN, frame_file)
    if match is None:
        print >>sys.stderr, 'slice path %s, ignoring frame %s' % (slice_path, frame_file)
        return None, metadata
    frame_path = os.path.join(slice_path, frame_file)
    d = pydicom.read_file(frame_path)
    frame = {
        'offset': int(match.group(1)),
        'time': int(match.group(2)),
        'pixel': pixels_from_dicom(d),
        'slice_thickness': float(d.SliceThickness),
        'scale_x': float(d.PixelSpacing[0]),
        'scale_y': float(d.PixelSpacing[1]),
        'slice_location': float(d.SliceLocation)
    }
    return frame, update_meta(metadata, d)


def update_meta(metadata, d):
    if d.PatientAge[-1] == 'Y':
        age = float(d.PatientAge[:-1])
    elif d.PatientAge[-1] == 'M':
        age = float(d.PatientAge[:-1]) / 12.
    elif d.PatientAge[-1] == 'W':
        age = float(d.PatientAge[:-1]) / 52.
    else:
        print >>sys.stderr, 'incorrect age %s' % d.PatientAge
    sex = str(d.PatientSex)
    if metadata.get('patient_age', age) != age:
        print >>sys.stderr, 'age does not match %s != %s' % (metadata['patient_age'], age)
    if metadata.get('patient_sex', sex) != sex:
        print >>sys.stderr, 'sex does not match %s != %s' % (metadata['patient_sex'], sex)
    metadata['patient_age'] = age
    metadata['patient_sex'] = sex
    return metadata


def match_slice_type(slice_folder):
    for slice_type, pattern in FOLDER_PATTERN.items():
        match = re.match(pattern, slice_folder)
        if match:
            return slice_type, match
    return None, None


def get_slices(study_id, study_path):
    slices = defaultdict(dict)
    metadata = dict()
    for slice_folder in os.listdir(study_path):
        slice_type, match = match_slice_type(slice_folder)
        if slice_type is None:
            print >>sys.stderr, 'study %s, ignoring slice %s' % (study_id, slice_folder)
            continue
        slice_id = int(match.group(1))
        slice_path = os.path.join(study_path, slice_folder)
        frames = []
        for frame_file in os.listdir(slice_path):
            frame, metadata = get_frame(slice_path, frame_file, metadata)
            if frame:
                frames.append(frame)
        if len(frames) > 0 and any(frames[0]['offset'] != f['offset'] for f in frames):
            print >>sys.stderr, 'offsets do not match for study %s, slice %s' % (study_id, slice_id)
        frames = sorted(frames, key=lambda f: f['time'])
        if frames:  # don't add empty slices
            slices[slice_type][slice_id] = frames
        else:
            print >>sys.stderr, 'empty slice for study %s, slice %s' % (study_id, slice_id)
    return metadata, slices


def get_data(folder_name):
    """Gets all studies from director (e.g. ./data/train)."""
    train_data = {}
    for study_id, study_path in sorted(get_studies(folder_name)):
        train_data[study_id] = get_slices(study_id, study_path)
    return train_data
