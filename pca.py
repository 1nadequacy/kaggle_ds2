"""PCA for images.

Adapted from http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html

"""
import numpy as np
import os
import pickle
import random

def pca(X, k=3):
    """Performs PCA.

    X: matrix with training data as flattened arrays in rows
    k: number of eigenvector, eigenvalues to return
    Returns projection matrix (with important dimensions first) and eigenvalues.

    """
    # get dimensions
    num_data, dim = X.shape

    # center data
    X -= X.mean(axis=0)

    M = np.dot(X, X.T) / X.shape[0]  # covariance matrix
    e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors
    tmp = np.dot(X.T, EV).T  # this is the compact trick
    V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
    eig = e[::-1]

    return V[:k], eig[:k]

def images_pca(images_folder, limit=100, k=3):
    """Performs PCA on randomly selected subset of images.

    images_folder: folder of studies
    limit: maximum number of images to select
    k: number of components for PCA

    """
    my_images = []
    shape = None
    files = os.listdir(images_folder)
    random.shuffle(files)
    files = files[:limit]
    for study_file in files:
        assert study_file.endswith('.pkl'), 'file %s has wrong extension' % study_file
        with open(os.path.join(images_folder, study_file), 'rb') as f:
            study = pickle.load(f)
            for slice_ in study['sax']:
                myframe = random.choice(study['sax'][slice_])
                assert shape is None or shape == myframe['pixel'].shape, 'inconsistent image shapes'
                shape = myframe['pixel'].shape
                my_images.append(myframe['pixel'])

    X = np.zeros((len(my_images), my_images[0].size))
    for i, img in enumerate(my_images):
        X[i] = img.reshape(img.size)

    V, eig = pca(X)
    V = V.reshape((k, shape[0], shape[1]))
    return V, eig

def pca_noise(V, eig, std_dev=0.1, normalize=True):
    """Returns randomly sampled noise based on PCA."""
    lambdas = eig / np.linalg.norm(eig) if normalize else eig

    k = len(lambdas)
    noise = np.random.normal(scale=std_dev, size=k)

    ret = (lambdas * noise).reshape((k, 1, 1)) * V
    return ret.sum(axis=0)
