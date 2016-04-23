import os
import numpy as np
from reikna.cluda import cuda_api, dtypes
from reikna.fft import FFT


class _FFTLocalState(object):
    def __init__(self):
        self._pid = None
        self._cluda_api = None
        self._thr = None
        self._fftc = None

    def _initialize(self):
        if self._pid != os.getpid():
            print 'Initializing CUDA thread for pid', os.getpid()
            self._cluda_api = cuda_api()
            self._thr = self._cluda_api.Thread.create()
            self._fftc = {}
            self._pid = os.getpid()

    def get_thread(self):
        self._initialize()
        return self._thr

    def get_fftc(self, arr):
        self._initialize()

        shape = arr.shape
        if shape in self._fftc:
            return self._fftc[shape]

        fft = FFT(self._thr.array(shape, np.complex64))
        fftc = fft.compile(self._thr)
        self._fftc[shape] = fftc
        return fftc

_local_state = _FFTLocalState()


def fft(arr, inverse=False):
    if arr.dtype != np.complex64:
        arr = arr.astype(np.complex64)
    fftc = _local_state.get_fftc(arr)
    arr_dev = _local_state.get_thread().to_device(arr)
    res_dev = _local_state.get_thread().array(arr.shape, np.complex64)
    fftc(res_dev, arr_dev, int(inverse))
    return res_dev.get()
