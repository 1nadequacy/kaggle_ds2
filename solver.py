"""Wrapper around Caffe's SGDSolver.

"""
import common

import caffe
import os
import numpy as np

SIGMAS = [22]


def train(solver_file, niter=1000, nbatches=1000, ntest_batches=100, sigmas=SIGMAS):
    assert os.path.exists(solver_file), 'solver file does not exist'
    solver = caffe.SGDSolver(solver_file)
    stats = {}

    for it in xrange(niter):
        print '>>>>>> Iteration', it, 'training', nbatches, 'batches'
        solver.step(nbatches)

        print '>>>>>> Iteration', it, 'testing on', ntest_batches, 'batches'
        crps = 0.0
        smooth = np.zeros(len(sigmas))
        loss = 0.0
        counter = 0.0
        for test_it in xrange(ntest_batches):
            solver.test_nets[0].forward()
            out = solver.test_nets[0].blobs['out'].data
            label = solver.test_nets[0].blobs['label'].data
            loss += float(solver.test_nets[0].blobs['loss'].data)
            for i in xrange(out.shape[0]):
                crps += common.crps_sample_sample(
                    label[i],
                    out[i])
                for sigma_id, sigma in enumerate(sigmas):
                    smooth[sigma_id] += common.crps_sample(
                        label[i],
                        common.smooth_cdf(out[i], sigma))
                counter += 1
        crps /= counter
        smooth /= counter
        loss /= ntest_batches
        stats[it] = {'crps' : crps, 'loss': loss, 'smooth': smooth}
        print '>>>>>> Test CRPS: %s, Test Loss: %s, Test smooth %s' % (crps, loss, smooth)

    return stats
