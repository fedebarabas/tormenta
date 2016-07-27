# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 20:51:54 2015

@author: Federico Barabas
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import affine_transform
import tifffile as tiff
import h5py as hdf
from tkinter import Tk, filedialog

from tormenta.analysis.maxima import Maxima

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def load_images(filename):
    try:
        return load_tiff(filename)
    except:
        return load_hdf(filename)


def load_tiff(filename):

    with tiff.TiffFile(filename) as ff:
        data = ff.asarray()
        if len(data) > 0:
            data = np.mean(data, 0)
        return split_images(data)


def load_hdf(filename):

    with hdf.File(filename, 'r') as ff:
        return split_images(np.mean(ff['data'].value, 0))


def split_images(arr):

    images = np.zeros((2, 128, 266), dtype=np.uint16)
    center = int(0.5*arr.shape[0])
    images[0] = arr[:center - 5, :]
    images[1] = arr[center + 5:, :]

    return images


def fit_and_plot(images, fig):
    points = []
    marks = ['rx', 'bs']
    for k in [0, 1]:
        mm = Maxima(images[k])
        mm.find(alpha=2.5)
        mm.getParameters()
        mm.fit()
        pp = np.zeros((len(mm.results['fit_x']), 2))
        pp[:, 0] = mm.results['fit_x']
        pp[:, 1] = mm.results['fit_y']
        points.append(pp)

        # Image plot
        ax = fig.add_subplot(311 + k)
        ax.imshow(mm.image, interpolation='None', aspect='equal',
                  cmap='cubehelix', vmin=0, vmax=700)
        ax.autoscale(False)
        ax.plot(mm.results['fit_y'] - 0.5, mm.results['fit_x'] - 0.5,
                marks[k], mew=1, ms=5, markerfacecolor='None')
        ax.set_adjustable('box-forced')

        for i in np.arange(len(mm.results['fit_y'])):
            ax.annotate(str(i), xy=(mm.results['fit_y'][i] - 0.1,
                                    mm.results['fit_x'][i] - 0.1))

    # superposition of channels plot
    ax = fig.add_subplot(313)
    ch_superposition(ax, points)
    plt.tight_layout()

    return points


def points_registration(images):
    """Points registration routine. It takes a tetraspeck image of each channel
    and calculates the affine transformation between them."""

    fig = plt.figure()
    fig.set_size_inches(7, 25, forward=True)
    points = fit_and_plot(images, fig)

    if len(points[0]) != len(points[1]):
        plt.show(block=False)
        points = remove_bad_points(points)

    plt.close()

    # Replotting images
    fig = plt.figure()
    plot_points(images, points, fig)
    plt.show(block=False)

    # points[1] must have the same order as points[0]
    order = input('Reorder points[1]: (ej: 0-1-4-3-2) ')
    try:
        order = list(map(int, [l for l in order.split('-')]))
        points[1] = points[1][order]
        plt.close()

        # Last check
        fig = plt.figure()
        plot_points(images, points, fig)
        plt.show()
    except:
        plt.close()

    return points


def plot_points(images, points, fig):

    marks = ['rx', 'bs']
    for k in [0, 1]:
        # Image plot
        ax = fig.add_subplot(311 + k)
        ax.imshow(images[k], interpolation='None', aspect='equal',
                  cmap='cubehelix', vmin=0, vmax=700)
        ax.autoscale(False)
        ax.plot(points[k][:, 1] - 0.5, points[k][:, 0] - 0.5,
                marks[k], mew=1, ms=5, markerfacecolor='None')
        ax.set_adjustable('box-forced')

        for i in np.arange(len(points[k])):
            ax.annotate(str(i), xy=(points[k][:, 1][i] - 0.5 - 0.1,
                                    points[k][:, 0][i] - 0.5 - 0.1))

    # superposition of channels plot
    ax = fig.add_subplot(313)
    ch_superposition(ax, points)
    plt.tight_layout()
    fig.set_size_inches(7, 25, forward=True)


def ch_superposition(ax, points):
    # superposition of channels plot
    ax.plot(points[0][:, 1] - 0.5, points[0][:, 0] - 0.5, 'rx', mew=1, ms=5)
    ax.plot(points[1][:, 1] - 0.5, points[1][:, 0] - 0.5, 'bs', mew=1, ms=5,
            markerfacecolor='None')

    for i in np.arange(len(points[0])):
        ax.annotate(str(i), xy=(points[0][:, 1][i] - 0.6,
                                points[0][:, 0][i] - 0.6), color='r')

    for i in np.arange(len(points[1])):
        ax.annotate(str(i), xy=(points[1][:, 1][i] - 5,
                                points[1][:, 0][i] - 1.6), color='b')

    ax.set_aspect('equal')
    ax.set_xlim(0, 266)
    ax.set_ylim(128, 0)


def remove_bad_points(points):

    for ch in [0, 1]:
        n = len(points[ch])
        random = int(np.random.rand()*n - 1)
        text = 'Bad points from channel {} (ej: 0-{}-{})'.format(ch, random, n)
        bpoints = input(text)
        if bpoints != '':
            bpoints = list(map(int, [l for l in bpoints.split('-')]))
            points[ch] = np.delete(points[ch], bpoints, 0)

    return points


def matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
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


def h_affine_transform(image, H):
    """ Transforms the image with the affine transformation matrix H.

    References:
    https://stackoverflow.com/questions/27546081/determining-a-homogeneous-
    affine-transformation-matrix-from-six-points-in-3d-usi/27547597#27547597
    http://elonen.iki.fi/code/misc-notes/affine-fit/
    """
    return affine_transform(image, H[:2, :2], (H[0, 2], H[1, 2]))


def matrix_from_stack(filename, Hfilename):

    images = load_images(filename)

    points = points_registration(images)
    H = matrix_from_points(points[0], points[1])
    print('Transformation matrix 1 --> 0')
    print(H)
    np.save(Hfilename, H)

    return H


def transformation_check(H, filename):
    """ Applies affine transformation H to the images in filename hdf5 file to
    check for the performance of H."""

    images = load_images(filename)

    images2 = np.zeros((2, 128, 266), dtype=np.uint16)
    images2[0] = images[0]
    images2[1] = h_affine_transform(images[1], H)

    points = points_registration(images2)

    it = np.arange(len(points[0]))
    dist = np.array([np.linalg.norm(points[0][i] - points[1][i]) for i in it])
    print('Mean distance: ', np.mean(dist))
    print('Maximum distance: ', np.max(dist))


def find_largest_rectangle(a):
    ''' Adapted from
    http://stackoverflow.com/questions/2478447/
    find-largest-rectangle-containing-only-zeros-in-an-n%C3%97n-binary-matrix

    Usage:
    s = ''0 0 0 0 1 0
    0 0 1 0 0 1
    0 0 0 0 0 0
    1 0 0 0 0 0
    0 0 0 0 0 1
    0 0 1 0 0 0''
    a = np.fromstring(s, dtype=int, sep=' ').reshape(6, 6)
    find_largest_rectangle(a)
    '''

    a = (1 - a).astype(np.int)

    area_max = (0, [])

    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            if a[r][c] == 1:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r-dh][c])
                area = (dh + 1)*minw
                if area > area_max[0]:
                    area_max = (area, [(r-dh, r), (c-minw + 1, c)])

#    print('area', area_max[0])
    xlim, ylim = area_max[1]

    return xlim, ylim


def get_affine_shapes(H):

    data = np.ones((128, 266))
    datac = h_affine_transform(data, H)

#    indices = np.where(datac == 1)
#
#    # This may only work with the present setup and two-color scheme
#    ylim = (indices[1].min() + 1, indices[1].max() + 1)
#    xmin = indices[0].min()
#    while True:
#        if np.sum(datac[xmin, ylim[0]:ylim[1]] == 0) == 0:
#            # If all the elements are ones
#            break
#        else:
#            xmin += 1
#
#    xmax = indices[0].max()
#    while True:
#        if np.sum(datac[xmax, ylim[0]:ylim[1]] == 0) == 0:
#            break
#        else:
#            xmax -= 1
#
#    xlim = (xmin, xmax + 1)
#    ylim = (indices[1].min(), indices[1].max() + 1)

    xlim, ylim = find_largest_rectangle(datac)
    cropShape = (xlim[1] - xlim[0], ylim[1] - ylim[0])

    return xlim, ylim, cropShape

if __name__ == '__main__':

    root = Tk()
    root.withdraw()
    types = [('tiff files', '.tiff'), ('hdf5 files', '.hdf5'),
             ('all files', '.*')]

    filename = filedialog.askopenfilename(filetypes=types, parent=root,
                                          title='Load bead stack')
    folder = os.path.split(filename)[0]

    arrayType = [('numpy array', '.npy')]
    Hfilename = filedialog.asksaveasfilename(filetypes=arrayType,
                                             parent=root, initialdir=folder,
                                             title='Save affine matrix')
    root.destroy()

    H = matrix_from_stack(filename, Hfilename)
    print('Checking transformation with same stack')
    transformation_check(H, filename)

    check = input('Check the transformation with another stack? (y/n) ') == 'y'
    if check:
        root = Tk()
        root.withdraw()
        title = 'Load stack to test the affine transformation H'
        filename = filedialog.askopenfilename(filetypes=types, parent=root,
                                              initialdir=folder, title=title)
        root.destroy()
        transformation_check(H, filename)
        input('Press any key to exit...')
