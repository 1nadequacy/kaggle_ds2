"""Caffe model definitions.

Each method takes in the locations of the LMDB data and LMDB labels as
well as batch size, returning a proto string of the model.

"""
import caffe
from caffe import layers as L
from caffe import params as P

import os

DEFAULT_SOLVER_PARAMS = {
    'test_iter': 10,
    'test_interval': 10,
    'base_lr': 0.001,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'lr_policy': '"step"',
    'gamma': 0.5,  # halve learning rate
    'power': 0.75,
    'stepsize': 2000,
    'display': 100,
    'max_iter': 100000,
    'snapshot': 5000}

def write_model(model_name, out_dir, model_fn,
                lmdb_train, lmdb_train_lab, lmdb_test, lmdb_test_lab,
                batch_size, *args, **kwargs):
    """Entry point for creating model for use by Caffe's solver.

    This method creates the model training and test protos, creates the
    default solver specifications, and saves all of these to files.

    """
    out_train = os.path.join(out_dir, '%s_train.prototxt' % model_name)
    out_test = os.path.join(out_dir, '%s_test.prototxt' % model_name)
    out_deploy = os.path.join(out_dir, '%s_deploy.prototxt' % model_name)
    out_solver = os.path.join(out_dir, '%s_solver.prototxt' % model_name)
    out_snapshot = os.path.join(out_dir, model_name)

    with open(out_train, 'w') as f:
        f.write(model_fn(lmdb_train, lmdb_train_lab, batch_size, False,
                             *args, **kwargs))

    with open(out_test, 'w') as f:
        f.write(model_fn(lmdb_test, lmdb_test_lab, batch_size, False,
                             *args, **kwargs))

    with open(out_deploy, 'w') as f:
        f.write(model_fn('', '', 1, True, *args, **kwargs))

    with open(out_solver, 'w') as f:
        print >>f, 'train_net:', '"%s"' % out_train
        print >>f, 'test_net:', '"%s"' % out_test
        print >>f
        print >>f, 'snapshot_prefix:', '"%s"' % out_snapshot
        print >>f
        for param, val in DEFAULT_SOLVER_PARAMS.items():
            print >>f, '%s:' % param, val

def lenet(lmdb_data, lmdb_label, batch_size, deploy, crop=64, mirror=False):
    """Simple LeNet to predict cdf."""
    data_transforms = dict(scale=1.)
    if crop:  # will crop images to [crop]x[crop] with random center
        data_transforms['crop_size'] = crop
    if mirror:  # will randomly flip images
        data_transforms['mirror'] = 1

    n = caffe.NetSpec()
    if deploy:
        input_ = "data"
        dim1 = batch_size
        dim2 = 3  # need to change these manually
        dim3 = 64
        dim4 = 64
        n.data=L.Layer()
    else:
        n.data = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                        source=lmdb_data, transform_param=data_transforms, ntop=1)
        n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                         source=lmdb_label, ntop=1)

    # first convolutional layer
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=40, weight_filler=dict(type='xavier'))
    n.norm1 = L.BatchNorm(n.conv1)
    n.relu1 = L.ReLU(n.norm1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # second convolutional layer
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=40, weight_filler=dict(type='xavier'))
    n.norm2 = L.BatchNorm(n.conv2)
    n.relu2 = L.ReLU(n.norm2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # fully connected layers
    n.drop = L.Dropout(n.pool2, dropout_ratio=0.5)
    n.ip1 = L.InnerProduct(n.drop, num_output=600, weight_filler=dict(type='xavier'))
    n.out = L.Sigmoid(n.ip1)
    if deploy:
        deploy_str = ('input: {}\ninput_dim: {}\n'
                      'input_dim: {}\ninput_dim: {}\n'
                      'input_dim: {}').format(
            '"%s"' % input_,
            dim1, dim2, dim3, dim4)
        return (deploy_str + '\n' + 'layer {' +
                'layer {'.join(str(n.to_proto()).split('layer {')[2:]))
    else:
        n.loss = L.EuclideanLoss(n.out, n.label)
        return str(n.to_proto())

def lenet_weight(lmdb_data, lmdb_label, batch_size, deploy, crop=64, mirror=False):
    """Simple LeNet to predict weight."""
    data_transforms = dict(scale=1.)
    if crop:  # will crop images to [crop]x[crop] with random center
        data_transforms['crop_size'] = crop
    if mirror:  # will randomly flip images
        data_transforms['mirror'] = 1

    n = caffe.NetSpec()
    if deploy:
        input_ = "data"
        dim1 = batch_size
        dim2 = 3  # need to change these manually
        dim3 = 64
        dim4 = 64
        n.data=L.Layer()
    else:
        n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB,
                        source=lmdb_data, transform_param=data_transforms, ntop=2)

    # first convolutional layer
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=40, weight_filler=dict(type='xavier'))
    n.norm1 = L.BatchNorm(n.conv1)
    n.relu1 = L.ReLU(n.norm1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # second convolutional layer
    n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=40, weight_filler=dict(type='xavier'))
    n.norm2 = L.BatchNorm(n.conv2)
    n.relu2 = L.ReLU(n.norm2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, kernel_size=2, stride=2, pool=P.Pooling.MAX)

    # fully connected layers
    n.drop = L.Dropout(n.pool2, dropout_ratio=0.5)
    n.ip1 = L.InnerProduct(n.drop, num_output=600, weight_filler=dict(type='xavier'))
    n.sig1 = L.Sigmoid(n.ip1)
    n.out = L.InnerProduct(n.sig1, num_output=1, weight_filler=dict(type='xavier'))
    if deploy:
        deploy_str = ('input: {}\ninput_dim: {}\n'
                      'input_dim: {}\ninput_dim: {}\n'
                      'input_dim: {}').format(
            '"%s"' % input_,
            dim1, dim2, dim3, dim4)
        return (deploy_str + '\n' + 'layer {' +
                'layer {'.join(str(n.to_proto()).split('layer {')[2:]))
    else:
        n.loss = L.EuclideanLoss(n.out, n.label)
        return str(n.to_proto())
