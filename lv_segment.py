"""Performs segmentation for left ventricle

Based on https://gist.github.com/ajsander/b65061d12f50de3cef5d#file-fcn_tutorial-ipynb

"""

import caffe
import common
import numpy as np
import scipy
import os

SUNNYBROOK_DATASET_MEAN = 77.0
HEART_RADIUS = 80  # in millimeters
MODEL_DATA = './lv_segmentation'
DEPLOY_PROTO = 'fcn_deploy.prototxt'
MODEL_SNAPSHOT = 'model_logs/fcn_iter_15000.caffemodel'


def calc_rois(study, desired_radius=HEART_RADIUS):
    """Given study returns tuple (masked_rois, circles).

    desired_radius is the desired radius of the ROI in mm.
    circles is a dict of slice_: (ctr_x, ctr_y) for each slice.

    """
    net = get_model()
    rois = []
    centers = dict()
    slice_numbers, slices = common.study_to_numpy(study, with_slice_number=True)
    for slice_number, slice_ in zip(slice_numbers, slices):
        masks = []
        for frame in slice_:
            masks.append(evaluate(net, frame))
        mean_mask = np.mean(np.array(masks), axis=0)
        ctr_x, ctr_y = map(int, scipy.ndimage.measurements.center_of_mass(mean_mask))
        rois.append(mean_mask)
        centers[slice_number] = (ctr_x, ctr_y)

    return np.array(rois), centers


def get_model():
    proto = os.path.join(MODEL_DATA, DEPLOY_PROTO)
    model = os.path.join(MODEL_DATA, MODEL_SNAPSHOT)
    net = caffe.Net(proto, model, caffe.TEST)
    return net


def evaluate(net, frame):
    pixel = frame.astype(np.float)
    pixel -= np.array([SUNNYBROOK_DATASET_MEAN])
    blob = np.expand_dims(pixel, axis=0)
    net.blobs['data'].reshape(1, *blob.shape)
    net.blobs['data'].data[...] = blob
    net.forward()
    # model outputs 2 layers, can take any of them
    prob = net.blobs['prob'].data[0][1]
    assert prob.shape == frame.shape, 'predicted mask has incorrect shape'
    return prob

