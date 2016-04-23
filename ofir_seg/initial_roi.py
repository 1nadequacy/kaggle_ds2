"""Get initial ROI from pre-trained NN."""

import numpy as np
import caffe
import scipy
from skimage import transform

proto = 'roi_deploy.prototxt'
model = './model_logs/roi_con_iter_3000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(proto, model, caffe.TEST)

def preproc_image(image):
    return (transform.resize(image, (64, 64))).astype(np.int)

def get_center(image):
    shape_x, shape_y = image.shape
    assert shape_x == shape_y
    input_ = preproc_image(image)
    input_ = np.expand_dims(np.expand_dims(input_, axis=0), axis=0)
    net.blobs['data'].data[...] = input_
    net.forward()
    out = net.blobs['ip1'].data.reshape((32, 32))

    ctr_x, ctr_y = scipy.ndimage.measurements.center_of_mass(out)
    return int(ctr_x * shape_x / 32), int(ctr_y * shape_y / 32)

def get_real_center(contour):
    return map(int, scipy.ndimage.measurements.center_of_mass(contour > 0))
