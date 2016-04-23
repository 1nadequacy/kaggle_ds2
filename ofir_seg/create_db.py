"""Run this script to create LevelDB from the raw data

Source: https://gist.github.com/ajsander/b65061d12f50de3cef5d#file-fcn_tutorial-ipynb

"""

import initial_roi

import os, fnmatch, shutil, subprocess, re, sys
import dicom, lmdb, cv2
import numpy as np
import caffe
import scipy
from skimage import transform, exposure
import random
import math

BASE_FOLDER = '/mnt/lv/'
IMAGE_FOLDER = os.path.join(BASE_FOLDER, 'data', 'challenge_training')
CONTOUR_FOLDER = os.path.join(BASE_FOLDER, 'labels',
    'Sunnybrook Cardiac MR Database ContoursPart3', 'TrainingDataContours')
CONTOUR_PATTERN = 'IM-0001-*-icontour-manual.txt'
SPLIT_RATIO = 0.1

SAX_SERIES = {
    # challenge training
    'SC-HF-I-1'  : '0004',
    'SC-HF-I-2'  : '0106',
    'SC-HF-I-4'  : '0116',
    'SC-HF-I-40' : '0134',
    'SC-HF-NI-3' : '0379',
    'SC-HF-NI-4' : '0501',
    'SC-HF-NI-34': '0446',
    'SC-HF-NI-36': '0474',
    'SC-HYP-1'   : '0550',
    'SC-HYP-3'   : '0650',
    'SC-HYP-38'  : '0734',
    'SC-HYP-40'  : '0755',
    'SC-N-2'     : '0898',
    'SC-N-3'     : '0915',
    'SC-N-40'    : '0944',
}

def get_all_contours(base_path):
    contours = []
    for dirpath, _, files in os.walk(base_path):
        for fl in fnmatch.filter(files, CONTOUR_PATTERN):
            contours.append(os.path.join(dirpath, fl))
    return contours


def get_image_by_contour(contour_path):
    pattern = r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt'
    match = re.search(pattern, contour_path)

    def shrink_zeros(study):
        def shrink_if_number(x):
            try:
                return str(int(x))
            except ValueError:
                return x
        return '-'.join([shrink_if_number(x) for x in study.split('-')])

    study, image = shrink_zeros(match.group(1)), match.group(2)
    image_name = 'IM-%s-%04d.dcm' % (SAX_SERIES[study], int(image))
    print IMAGE_FOLDER, study, image_name
    return os.path.join(IMAGE_FOLDER, study, image_name)


def random_transformation(img1, img2):
    shape_x, shape_y = img1.shape
    rot = (random.random() - 0.5) * math.pi / 4
    trans_x = int((random.random() - 0.5) * shape_x / 8)
    trans_y = int((random.random() - 0.5) * shape_y / 8)
    scale = 1. / 1.1 + random.random() * (1.1 - 1. / 1.1)
    pixel_scale = 1. / 1.1 + random.random() * (1.1 - 1. / 1.1)

    trans = transform.SimilarityTransform(
        scale=scale, rotation=rot, translation=(trans_x, trans_y))
    return \
        (pixel_scale * transform.warp(img1.astype(float), trans, mode='nearest')), \
        (transform.warp(img2.astype(float), trans, mode='nearest'))


def load_image_and_contour(contour_path, radius=None):
    image_path = get_image_by_contour(contour_path)
    f = dicom.read_file(image_path)
    window_min = int(f.WindowCenter) - 0.5 * int(f.WindowWidth)
    image = 255 * np.clip(
        (f.pixel_array.astype(np.float) - window_min) / float(f.WindowWidth),
        0.0, 1.0)
    contour = np.zeros_like(image, dtype='uint8')
    ctrs = np.loadtxt(contour_path, delimiter=' ').astype(np.int)
    cv2.fillPoly(contour, [ctrs], 1)

    def center(args):
        img, con = args
        ctr_x, ctr_y = initial_roi.get_real_center(con)
        ctr_x += radius * random.random() - radius // 2
        ctr_y += radius * random.random() - radius // 2
        def crop_ctr(ii):
            shape_x, shape_y = ii.shape
            return np.pad(ii, radius, mode='edge') \
                [ctr_x:ctr_x + 2 * radius, ctr_y:ctr_y + 2 * radius]
        return map(crop_ctr, (img, con))

    return map(center,
               [random_transformation(image, contour)
                for _ in xrange(10)])


def preproc_image(image):
    return (transform.resize(image, (64, 64))).astype(np.int)


def preproc_contour(contour, just_roi=True):
    shape_x, shape_y = contour.shape
    roi_radius = 50 * 32 / min(shape_x, shape_y)
    if just_roi:
        ctr_x, ctr_y = scipy.ndimage.measurements.center_of_mass(contour > 0)
        if np.isnan(ctr_x):
            print 'nan x'
            ctr_x = shape_x // 2
        if np.isnan(ctr_y):
            print 'nan y'
            ctr_y = shape_y // 2
    contour = transform.resize(contour.astype(float), (64, 64))
    if not just_roi:
        return (contour > 0).astype('uint8')
    ctr_x, ctr_y = int(32 * ctr_x / shape_x), int(32 * ctr_y / shape_y)
    ret = np.zeros((32, 32), dtype='uint8')
    cv2.circle(ret, center=(ctr_y, ctr_x), radius=roi_radius, color=1, thickness=-1)
    return ret


def save_to_db(data, image_db_name, contour_db_name):
    for db_name in [image_db_name, contour_db_name]:
        db_path = os.path.abspath(db_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

    image_db = lmdb.open(image_db_name, map_size=1e12)
    contour_db = lmdb.open(contour_db_name, map_size=1e12)
    with image_db.begin(write=True) as image_page:
        with contour_db.begin(write=True) as contour_page:
            for i, sample in enumerate(data):
                image, contour = sample
                image = preproc_image(image)
                contour = preproc_contour(contour, just_roi=False)
                image_datum = caffe.io.array_to_datum(np.expand_dims(image, axis=0))
                contour_datum = caffe.io.array_to_datum(np.expand_dims(contour, axis=0))
                image_page.put('{:0>10d}'.format(i), image_datum.SerializeToString())
                contour_page.put('{:0>10d}'.format(i), contour_datum.SerializeToString())
                if i % 100 == 0:
                    print 'data', i
                    print np.mean(image)
                    print np.sum(image)
                    print np.mean(contour)
                    print np.sum(contour)

    image_db.close()
    contour_db.close()


def main():
    random.seed(888)
    np.random.seed(888)

    data = []
    for contour_path in get_all_contours(CONTOUR_FOLDER):
        data.extend(load_image_and_contour(contour_path, radius=32))
    
    print 'total number of examples: %s' % len(data)

    np.random.shuffle(data)
    test_size = int(SPLIT_RATIO * len(data))
    test = data[:test_size]
    train = data[test_size:]
    print 'train size: %s' % len(train)
    print 'test size: %s' % len(test)
    save_to_db(train, 'train_image_lmdb', 'train_contour_lmdb')
    save_to_db(test, 'test_image_lmdb', 'test_contour_lmdb')


if __name__ == '__main__':
    main()
