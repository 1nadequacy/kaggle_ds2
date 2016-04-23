"""Determines general region of heart by searching for periodic palpitations.

Based on https://www.kaggle.com/c/second-annual-data-science-bowl/details/fourier-based-tutorial

"""
import common

import cv2
import numpy as np
from scipy.fftpack import fftn, ifftn
from scipy.optimize import curve_fit
from scipy.spatial.distance import euclidean
from scipy.stats import linregress

STD_MULTIPLIER = 2
NUM_BINS = 100
HEART_RADIUS = 80  # in millimeters

def calc_rois(study, slice_type, desired_radius=HEART_RADIUS):
    """Given study returns tuple (masked_rois, circles).

    desired_radius is the desired radius of the ROI in mm.
    circles is a dict of slice_: (ctr_x, ctr_y) for each slice.

    """
    slice_numbers, slices = common.study_to_numpy(study, slice_type=slice_type, with_slice_number=True)
    slice_locations = common.slice_locations(study, slice_type)  # will be used as z-values
    z_values = [slice_locations[0]]
    for i in xrange(len(slice_locations) - 1):
        z_values.append(z_values[-1] + slice_locations[i + 1] - slice_locations[i])
        if z_values[-1] - z_values[-2] < 1:
            z_values[-1] = z_values[-2] + 1  # don't want a floats to blow up

    metadata = common.metadata(study, slice_type)
    scale_x, scale_y = metadata['scale_x'], metadata['scale_y']
    assert scale_x == scale_y, 'only support equal scaling currently'
    roi_radius = int(desired_radius / scale_x + 1)  # radius in pixels
    num_slices, _, shape_x, shape_y = slices.shape

    # original mean over time of slices
    dc = np.mean(slices, 1)

    # detect movements
    h1s = np.array([get_harmonic(slices[i]) for i in xrange(num_slices)])
    # remove noise
    m = np.max(h1s) * 0.05
    h1s[h1s < m] = 0

    # perform ROI-detection on all slices together
    centroid, regressed_h1s = regression_filter(h1s)

    # regress 3d line through centroid of each slice
    centroids = np.array([get_centroid(img) for img in regressed_h1s])
    coords = regress_centroids(centroids, z_values=z_values)

    # yield ROIs
    rois, circles = get_ROIs(dc, regressed_h1s, coords, z_values,
                             radius=roi_radius)

    # code uses x, y as in graphics, so put it back to row/col
    circles = dict((slice_number, (circles[i][0][1], circles[i][0][0]))
                   for i, slice_number in enumerate(slice_numbers))

    return rois, circles

def get_harmonic(frames):
    ff = fftn(frames)
    first_harmonic = ff[1, :, :]
    result = np.absolute(ifftn(first_harmonic))
    result = cv2.GaussianBlur(result, (5, 5), 0)
    return result

def regression_filter(imgs):
    """Iteratively removes points that are far from weighted-mean location."""

    condition = True
    centroid = None
    while condition:
        prev_centroid = centroid
        imgs = regress_and_filter_distant(imgs)
        centroid = get_centroid(imgs)
        condition = (prev_centroid is None or
                     np.linalg.norm(centroid - prev_centroid) > 1.0)
    return centroid, imgs

def get_circle_mask(shape_x, shape_y, ctr_x, ctr_y, radius):
    mask = np.zeros((shape_x, shape_y))
    cv2.circle(mask, center=(ctr_y, ctr_x),  # cv2 looks at array as image
               radius=radius, color=1,
                thickness=-1)  # -1 to fill it
    return mask

def get_centroid(img):
    """Returns weighted-centroid.

    Note: input may be any dimension.

    """
    nz = np.nonzero(img)
    points = np.transpose(nz)
    weights = img[nz]
    centroid = np.average(points, axis=0, weights=weights)
    return centroid

def regress_and_filter_distant(imgs, inplace=False):
    # first get centroid for each individual slice
    centroids = np.array([get_centroid(img) for img in imgs])
    raw_coords = np.transpose(np.nonzero(imgs))
    # fit 3d line through centroids
    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    # find and remove outliers
    (coords, dists, weights) = get_weighted_distances(
            imgs, raw_coords, xslope, xintercept, yslope, yintercept)
    outliers = get_outliers(coords, dists, weights)
    if inplace:
        imgs_cpy = imgs
    else:
        imgs_cpy = np.copy(imgs)
    for c in outliers:
        (z, x, y) = c
        imgs_cpy[z, x, y] = 0
    return imgs_cpy

def regress_centroids(cs, z_values=None):
    num_slices = len(cs)
    y_centroids = cs[:, 0]
    x_centroids = cs[:, 1]
    z_values = z_values or np.arange(num_slices)

    if num_slices > 1:
        (xslope, xintercept, _, _, _) = linregress(z_values, x_centroids)
        (yslope, yintercept, _, _, _) = linregress(z_values, y_centroids)
    elif num_slices == 1:
        xslope = yslope = 0.0
        xintercept = x_centroids[0]
        yintercept = y_centroids[0]
    else:
        xslope = yslope = xintercept = yintercept = 0.0

    return (xslope, xintercept, yslope, yintercept)

def get_weighted_distances(imgs, coords, xs, xi, ys, yi):
    a = np.array([0, yi, xi])  # intercept
    n = np.array([1, ys, xs])  # slope

    zeros = np.zeros(3)

    def dist(p):
        to_line = (a - p) - (np.dot((a - p), n) * n)
        d = euclidean(zeros, to_line)
        return d

    def weight(p):
        (z, y, x) = p
        return imgs[z, y, x]

    dists = np.array([dist(c) for c in coords])
    weights = np.array([weight(c) for c in coords])
    return (coords, dists, weights)

def get_outliers(coords, dists, weights):
    fivep = int(len(weights) * 0.05)
    ctr = 1
    while True:
        (mean, std, fn) = gaussian_fit(dists, weights)
        low_values = dists < (mean - STD_MULTIPLIER * np.abs(std))
        high_values = dists > (mean + STD_MULTIPLIER * np.abs(std))
        outliers = np.logical_or(low_values, high_values)
        if len(coords[outliers]) == len(coords):
            weights[-fivep * ctr:] = 0
            ctr += 1
        else:
            return coords[outliers]

def gaussian_fit(dists, weights):
    # based on http://stackoverflow.com/questions/11507028/fit-a-gaussian-function
    (x, y) = histogram_transform(dists, weights)
    fivep = int(len(x) * 0.05)
    xtmp = x
    ytmp = y
    fromFront = False
    while True:
        if len(xtmp) == 0 and len(ytmp) == 0:
            if fromFront:
                # well we failed
                idx = np.argmax(y)
                xmax = x[idx]
                p0 = [max(y), xmax, xmax]
                (A, mu, sigma) = p0
                return mu, sigma, lambda x: gauss(x, A, mu, sigma)
            else:
                fromFront = True
                xtmp = x
                ytmp = y

        idx = np.argmax(ytmp)
        xmax = xtmp[idx]

        def gauss(x, *p):
            A, mu, sigma = p
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))

        p0 = [max(ytmp), xmax, xmax]
        try:
            coeff, var_matrix = curve_fit(gauss, xtmp, ytmp, p0=p0)
            (A, mu, sigma) = coeff
            return (mu, sigma, lambda x: gauss(x, A, mu, sigma))
        except RuntimeError:
            if fromFront:
                xtmp = xtmp[fivep:]
                ytmp = ytmp[fivep:]
            else:
                xtmp = xtmp[:-fivep]
                ytmp = ytmp[:-fivep]

def histogram_transform(values, weights):
    hist, bins = np.histogram(values, bins=NUM_BINS, weights=weights)
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + (bin_width / 2)

    return (bin_centers, hist)

def post_process_regression(imgs, inplace=False):
    (numimgs, _, _) = imgs.shape
    centroids = np.array([get_centroid(img) for img in imgs])

    (xslope, xintercept, yslope, yintercept) = regress_centroids(centroids)
    if inplace:
        imgs_cpy = imgs
    else:
        imgs_cpy = np.copy(imgs)

    def filter_one_img(zlvl):
        points_on_zlvl = np.transpose(imgs[zlvl].nonzero())
        points_on_zlvl = np.insert(points_on_zlvl, 0, zlvl, axis=1)
        (coords, dists, weights) = get_weighted_distances(
                imgs, points_on_zlvl, xslope, xintercept, yslope, yintercept)
        outliers = get_outliers(coords, dists, weights)
        for c in outliers:
            (z, x, y) = c
            imgs_cpy[z, x, y] = 0

    for z in range(numimgs):
        filter_one_img(z)

    return (imgs_cpy, (xslope, xintercept, yslope, yintercept))


def get_ROIs(originals, h1s, regression_params, z_values, radius=None):
    (xslope, xintercept, yslope, yintercept) = regression_params
    results = []
    circles = []
    for i, z in enumerate(z_values):
        o = originals[i]
        h = h1s[i]
        ctr = (xintercept + xslope * z, yintercept + yslope * z)
        r = radius or circle_smart_radius(h, ctr)
        tmp = np.zeros_like(o)
        floats_draw_circle(tmp, ctr, r, 1, -1)
        results.append(tmp * o)
        circles.append((ctr, r))

    return (np.array(results), np.array(circles))

def circle_smart_radius(img, center):
    domain = np.arange(1, 100)
    (xintercept, yintercept) = center

    def ratio(r):
        return filled_ratio_of_circle(img, (xintercept, yintercept), r) * r

    y = np.array([ratio(d) for d in domain])
    most = np.argmax(y)
    return domain[most]

def filled_ratio_of_circle(img, center, r):
    mask = np.zeros_like(img)
    floats_draw_circle(mask, center, r, 1, -1)
    masked = mask * img
    (x, _) = np.nonzero(mask)
    (x2, _) = np.nonzero(masked)
    if x.size == 0:
        return 0
    return float(x2.size) / x.size

def floats_draw_circle(img, center, r, color, thickness):
    (x, y) = center
    x, y = int(np.round(x)), int(np.round(y))
    r = int(np.round(r))
    cv2.circle(img, center=(x, y), radius=r, color=color, thickness=thickness)
