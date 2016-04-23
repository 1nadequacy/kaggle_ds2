"""Get contour from ROI."""

import numpy as np
import caffe
import scipy
from skimage import transform
import pickle

proto = 'decon_deploy.prototxt'
model = 'model_logs/decon_iter_15000.caffemodel'
caffe.set_mode_gpu()
net = caffe.Net(proto, model, caffe.TEST)

kaggle_cdf = pickle.load(open('kaggle_cdf.pkl', 'rb'))
sunnybrook_cdf = pickle.load(open('sunnybrook_cdf.pkl', 'rb'))

def preproc_image(image, normalize):
    if normalize:
        image = np.interp(image, kaggle_cdf[1], kaggle_cdf[0])
        image = np.interp(image, sunnybrook_cdf[0], sunnybrook_cdf[1])
    return (transform.resize(image, (64, 64)))

def get_contour(image, normalize=True):
    shape_x, shape_y = image.shape
    assert shape_x == shape_y
    input_ = preproc_image(image, normalize)
    input_ = np.expand_dims(np.expand_dims(input_, axis=0), axis=0)
    net.blobs['data'].data[...] = input_
    net.forward()
    out = net.blobs['out'].data.reshape((64, 64))
    return out
