import caffe
from caffe import layers as L
from caffe import params as P


def construct_fcn(image_lmdb, contour_lmdb, batch_size=1, include_acc=False):
    net = caffe.NetSpec()

    # args for convlution layers
    weight_filler = dict(type='gaussian', mean=0.0, std=0.01)
    bias_filler = dict(type='constant', value=0.1)
    param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]

    net.data = L.Data(source=image_lmdb, backend=P.Data.LMDB, batch_size=batch_size,
        ntop=1, transform_param=dict(crop_size=0, mean_value=[77], mirror=False))
    net.label = L.Data(source=contour_lmdb, backend=P.Data.LMDB,
        batch_size=batch_size, ntop=1)
    # conv-relu-pool 1
    net.conv1 = L.Convolution(net.data, kernel_size=5, stride=2, num_output=100,
        pad=50, group=1, weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    net.relu1 = L.ReLU(net.conv1, in_place=True)
    net.pool1 = L.Pooling(net.relu1, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    # conv-relu-pool 2
    net.conv2 = L.Convolution(net.pool1, kernel_size=5, stride=2, num_output=200,
        pad=0, group=1, weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    net.relu2 = L.ReLU(net.conv2, in_place=True)
    net.pool2 = L.Pooling(net.relu2, pool=P.Pooling.MAX, kernel_size=2, stride=2)
    net.conv3 = L.Convolution(net.pool2, kernel_size=3, stride=1, num_output=300,
        pad=0, group=1, weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    net.relu3 = L.ReLU(net.conv3, in_place=True)
    net.conv4 = L.Convolution(net.relu3, kernel_size=3, stride=1, num_output=300,
        pad=0, group=1, weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    net.relu4 = L.ReLU(net.conv4, in_place=True)
    net.drop = L.Dropout(net.relu4, dropout_ratio=0.1, in_place=True)
    net.score_classes = L.Convolution(net.drop, kernel_size=1, stride=1, num_output=2,
        pad=0, group=1, weight_filler=weight_filler, bias_filler=bias_filler, param=param)
    net.upscore = L.Deconvolution(net.score_classes)
    net.score = L.Crop(net.upscore, net.data)
    net.loss = L.SoftmaxWithLoss(net.score, net.label, loss_param=dict(normalize=True))
    if include_acc:
        net.accuracy = L.Accuracy(net.score, net.label)

    return net.to_proto()


def main():
    header = 'name: "fcn"\nforce_backward: true\n'

    with open('fcn_train2.prototxt', 'w') as f:
        net_proto = construct_fcn(
            'train_image_lmdb/',
            'train_contour_lmdb/')
        f.write(header + str(net_proto))

    with open('fcn_test2.prototxt', 'w') as f:
        net_proto = construct_fcn(
            'test_image_lmdb/',
            'test_contour_lmdb/',
            include_acc=True)
        f.write(header + str(net_proto))


if __name__ == '__main__':
    main()

