import caffe
from caffe import layers as L
from caffe import params as P


def construct(image_lmdb, contour_lmdb, batch_size=1):
    net = caffe.NetSpec()

    weight_filler = dict(type='xavier')
    bias_filler = dict(type='constant', value=0.0)
    param = [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)]

    net.data = L.Data(source=image_lmdb, backend=P.Data.LMDB, 
                      batch_size=batch_size, ntop=1, 
                      transform_param=dict(crop_size=0, mean_value=[0], mirror=False))
    net.label = L.Data(source=contour_lmdb, backend=P.Data.LMDB,
                       batch_size=batch_size, ntop=1)

    net.conv1 = L.Convolution(net.data, kernel_size=5, stride=1, num_output=100,
                              pad=0, group=1, weight_filler=weight_filler,
                              bias_filler=bias_filler, param=param)
    net.norm1 = L.BatchNorm(net.conv1)
    net.relu1 = L.ReLU(net.norm1, in_place=True)
    net.pool1 = L.Pooling(net.relu1, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2 = L.Convolution(net.pool1, kernel_size=4, stride=1, num_output=100,
                              pad=0, group=1, weight_filler=weight_filler,
                              bias_filler=bias_filler, param=param)
    net.norm2 = L.BatchNorm(net.conv2)
    net.relu2 = L.ReLU(net.norm2, in_place=True)
    net.pool2 = L.Pooling(net.relu2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3 = L.Convolution(net.pool2, kernel_size=3, stride=1, num_output=100,
                              pad=0, group=1, weight_filler=weight_filler,
                              bias_filler=bias_filler, param=param)
    net.norm3 = L.BatchNorm(net.conv3)
    net.relu3 = L.ReLU(net.norm3, in_place=True)
    net.pool3 = L.Pooling(net.relu3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.drop = L.Dropout(net.pool3, dropout_ratio=0.2, in_place=True)
    net.ip1 = L.InnerProduct(net.drop, num_output=32 * 32,
                             weight_filler=weight_filler)
    
    net.out = L.Sigmoid(net.ip1, in_place=True)
    net.flat_label = L.Flatten(net.label)
    net.loss = L.EuclideanLoss(net.out, net.flat_label)

    return net.to_proto()


def main():
    header = 'name: "fcn"\nforce_backward: true\n'

    with open('roi_train.prototxt', 'w') as f:
        net_proto = construct(
            'train_image_lmdb/',
            'train_contour_lmdb/',
            batch_size=16)
        f.write(header + str(net_proto))

    with open('roi_test.prototxt', 'w') as f:
        net_proto = construct(
            'test_image_lmdb/',
            'test_contour_lmdb/')
        f.write(header + str(net_proto))


if __name__ == '__main__':
    main()
