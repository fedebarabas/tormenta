# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:01 2014

@author: fbaraba
"""
import math
import numpy as np
from scipy.ndimage import affine_transform
from scipy.special import jn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def gaussian(x, fwhm):
    return np.exp(- 4 * np.log(2) * (x / fwhm)**2)


def best_gauss(x, x0, fwhm):
    """ Returns the closest gaussian function to an Airy disk centered in x0
    and a full width half maximum equal to fwhm."""
    return np.exp(- 4 * np.log(2) * (x - x0)**2 / fwhm**2)


def airy(x):
    return (2 * jn(1, 2 * np.pi * x) / (2 * np.pi * x))**2


def get_fwhm(wavelength, NA):
    ''' Gives the FWHM (in nm) for a PSF with wavelength in nm'''

    x = np.arange(-2, 2, 0.01)
    y = airy(x)

    # Fitting only inside first Airy's ring
    fit_int = np.where(abs(x) < 0.61)[0]

    fit_par, fit_var = curve_fit(gaussian, x[fit_int], y[fit_int], p0=0.5)

    return fit_par[0] * wavelength / NA


def airy_vs_gauss():

    wavelength = 670        # nm
    NA = 1.4

    x = np.arange(-2, 2, 0.01)
    y = airy(x)
    fw = get_fwhm(wavelength, NA)
    fit = best_gauss(x, 0, fw * NA / wavelength)

    print('FWHM is', np.round(fw))

    plt.plot(x, y, label='Airy disk')
    plt.plot(x, fit, label='Gaussian fit')
    plt.legend()
    plt.grid('on')
    plt.show()


def mode(array):
    hist, bin_edges = np.histogram(array, bins=array.max() - array.min())
    hist_max = hist.argmax()
    return (bin_edges[hist_max + 1] + bin_edges[hist_max]) / 2


def overlaps(p1, p2, d):
    return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) <= d


def dropOverlapping(maxima, d):
    """We exclude from the analysis all the maxima that have their fitting
    windows overlapped, i.e., the distance between them is less than 'd'
    """

    noOverlaps = np.zeros(maxima.shape, dtype=int)  # Final array

    n = 0
    for i in np.arange(len(maxima)):
        def overlapFunction(x):
            return not(overlaps(maxima[i], x, d))
        overlapsList = map(overlapFunction, np.delete(maxima, i, 0))
        if all(overlapsList):
            noOverlaps[n] = maxima[i]
            n += 1

    return noOverlaps[:n]


def kernel(fwhm):
    """ Returns the kernel of a convolution used for finding objects of a
    full width half maximum fwhm in an image."""
    window = np.ceil(fwhm) + 3
#    window = int(np.ceil(fwhm)) + 2
    x = np.arange(0, window)
    y = x
    xx, yy = np.meshgrid(x, y, sparse=True)
    matrix = best_gauss(xx, x.mean(), fwhm) * best_gauss(yy, y.mean(), fwhm)
    matrix /= matrix.sum()
    return matrix


def xkernel(fwhm):
    window = np.ceil(fwhm) + 3
    x = np.arange(0, window)
    matrix = best_gauss(x, x.mean(), fwhm)
    matrix = matrix - matrix.sum() / matrix.size
    return matrix


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
    """Return affine transform matrix to register two point sets.

    v0 and v1 are shape (\*, ndims) arrays of at least ndims non-homogeneous
    coordinates, where ndims is the dimensionality of the coordinate space.

    If shear is False, a similarity transformation matrix is returned.
    If also scale is False, a rigid/Euclidean transformation matrix
    is returned.

    By default the algorithm by Hartley and Zissermann [15] is used.
    If usesvd is True, similarity and Euclidean transformation matrices
    are calculated by minimizing the weighted sum of squared deviations
    (RMSD) according to the algorithm by Kabsch [8].
    Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
    is used, which is slower when using this Python implementation.

    The returned matrix performs rotation, translation and uniform scaling
    (if specified).

    >>> v0 = [[0, 1031, 1031, 0], [0, 0, 1600, 1600]]
    >>> v1 = [[675, 826, 826, 677], [55, 52, 281, 277]]
    >>> affine_matrix_from_points(v0, v1)
    array([[   0.14549,    0.00062,  675.50008],
           [   0.00048,    0.14094,   53.24971],
           [   0.     ,    0.     ,    1.     ]])
    >>> T = translation_matrix(np.random.random(3)-0.5)
    >>> R = random_rotation_matrix(np.random.random(3))
    >>> S = scale_matrix(random.random())
    >>> M = concatenate_matrices(T, R, S)
    >>> v0 = (np.random.rand(4, 100) - 0.5) * 20
    >>> v0[3] = 1
    >>> v1 = np.dot(M, v0)
    >>> v0[:3] += np.random.normal(0, 1e-8, 300).reshape(3, -1)
    >>> M = affine_matrix_from_points(v0[:3], v1[:3])
    >>> np.allclose(v1, np.dot(M, v0))
    True

    More examples in superimposition_matrix()

    Taken from http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    July 23th 2015

    (8)  A discussion of the solution for the best rotation to relate two sets
     of vectors. W Kabsch. Acta Cryst. 1978. A34, 827-828.
    (9)  Closed-form solution of absolute orientation using unit quaternions.
    (15) Multiple View Geometry in Computer Vision. Hartley and Zissermann.
     Cambridge University Press; 2nd Ed. 2004. Chapter 4, Algorithm 4.7, p 130.


    """
    v0 = np.array(np.transpose(v0), dtype=np.float64, copy=True)
    v1 = np.array(np.transpose(v1), dtype=np.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -np.mean(v0, axis=1)
    M0 = np.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -np.mean(v1, axis=1)
    M1 = np.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)

    if shear:
        # Affine transformation
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = np.dot(C, np.linalg.pinv(B))
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,)*ndims) + (1.0,)))
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))
        # rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)
        if np.linalg.det(R) < 0.0:
            # R does not constitute right handed system
            R -= np.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
            s[-1] *= -1.0
        # homogeneous transformation matrix
        M = np.identity(ndims+1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= np.sqrt(np.sum(v1) / np.sum(v0))

    # move centroids back
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))
    M /= M[ndims, ndims]
    return M


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [0.0,                 0.0,                 0.0, 1.0]])


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. Euclidean norm, of ndarray along axis.

    >>> v = np.random.random(3)
    >>> n = vector_norm(v)
    >>> np.allclose(n, np.linalg.norm(v))
    True
    >>> v = np.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> v = np.random.rand(5, 4, 3)
    >>> n = np.empty((5, 3))
    >>> vector_norm(v, axis=1, out=n)
    >>> np.allclose(n, np.sqrt(np.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def homo_affine_transform(image, H):
    """ Transforms the image with the affine transformation matrix H.

    References:
    https://stackoverflow.com/questions/27546081/determining-a-homogeneous-
    affine-transformation-matrix-from-six-points-in-3d-usi/27547597#27547597
    http://elonen.iki.fi/code/misc-notes/affine-fit/
    """
    return affine_transform(image, H[:2, :2], (H[0, 2], H[1, 2]))
