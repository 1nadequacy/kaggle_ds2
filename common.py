import sys
import numpy as np
import matplotlib.pyplot as plot
import pickle
from scipy.stats import norm


def slice_to_numpy(slice_):
    return np.array([f['pixel'] for f in slice_])


def study_to_numpy(study, slice_type, with_slice_number=False, unique_locations=False):
    data = study[slice_type]
    slices = []
    dim_x = 0
    dim_y = 0
    locations = set()
    for s, slice_ in data.items():
        frames = slice_to_numpy(slice_)
        location = int(slice_[0]['slice_location'] + 0.1)
        if unique_locations and location in locations:
            continue
        locations.add(location)
        slices.append(((slice_[0]['slice_location'], s), frames))
        dim_x = max(dim_x, slices[-1][-1].shape[1])
        dim_y = max(dim_y, slices[-1][-1].shape[2])

    # pad to be same dimensions
    for i, (_, slice_) in enumerate(slices):
        _, shape_x, shape_y = slice_.shape
        pad_x = ((dim_x - shape_x) // 2, (dim_x - shape_x + 1) // 2)
        pad_y = ((dim_y - shape_y) // 2, (dim_y - shape_y + 1) // 2)
        slices[i] = (slices[i][0], np.pad(slice_, [(0, 0), pad_x, pad_y], mode='edge'))

    # sort by slice location ...or otherwise by slice number
    if with_slice_number:
        return ([s for (_, s), _ in sorted(slices)],
                np.array([slice_ for _, slice_ in sorted(slices)]))
    else:
        return np.array([slice_ for _, slice_ in sorted(slices)])


def slice_locations(study, slice_type):
    data = study[slice_type]
    slice_locations = []
    for slice_ in data.values():
        slice_locations.append(slice_[0]['slice_location'])

    # sort by slice location
    return sorted(slice_locations)


def center_slices(study, slice_type):
    data = study[slice_type]
    if not data:
        return []
    slice_locations = []
    for slice_ in data.values():
        location = int(slice_[0]['slice_location'] + 0.1)
        slice_locations.append(location)
    slice_locations = sorted(set(slice_locations))
    center_location = slice_locations[len(slice_locations) // 2]

    return [s for s in data
            if int(data[s][0]['slice_location'] + 0.1) == center_location]


def metadata(study, slice_type):
    frame = study[slice_type].values()[0][0]
    ret = {'study': study['study'],
           'patient_age': study['patient_age'],
           'patient_sex': study['patient_sex'],
           'slice_thickness': frame['slice_thickness'],
           'scale_x': frame['scale_x'],
           'scale_y': frame['scale_y']}

    # consistency check (for some datasets this does occur)
    for slice_ in study[slice_type]:
        for frame in study[slice_type][slice_]:
            if frame['scale_x'] != ret['scale_x'] or frame['scale_y'] != ret['scale_y']:
                print >>sys.stderr, 'inconsistency in scales for slice %d: (%f, %f) vs (%f, %f)' \
                        % (slice_, ret['scale_x'], ret['scale_y'], frame['scale_x'], frame['scale_y'])

    return ret


def plot_image(img):
    plot.imshow(img)
    plot.show()


def plot_cdf(cdf):
    cdf = np.array(cdf)
    plot.plot(np.arange(cdf.shape[0]), cdf)


def crps_sample(vol, dist):
    assert len(dist) == 600
    s = 0.0
    cdf = 0.0
    for i in xrange(600):
        cdf = min(1.0, max(cdf, dist[i]))
        diff = cdf - (1.0 if i + 1e-9 > vol else 0.0)
        s += diff * diff
    return s / 600.0


def crps_sample_sample(vol, pred):
    return crps_sample(vol, (pred <= np.arange(600)).astype(float))


def crps_nparray_sample(vol, dist):
    s = 0.0
    cdf = 0.0
    for i in xrange(600):
        cdf = min(1.0, max(cdf, dist[i]))
        diff = vol[i] - cdf
        s += diff * diff
    return s / 600.0


def load_study_from_pickle(study_file):
    with open(study_file, 'rb') as f:
        return pickle.load(f)


def smooth_cdf(vol, sigma=0.1):
    return norm.cdf(np.linspace(0, 599, 600), vol, sigma)
