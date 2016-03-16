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

from tormenta.analysis.maxima import Maxima

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def load_tiff(filename):

    with tiff.TiffFile(filename) as ff:
        return split_images(ff.asarray())


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
    k = 311
    points = []
    for im in images:
        mm = Maxima(im)
        mm.find(alpha=2.5)
        mm.getParameters()
        mm.fit()
        pp = np.zeros((len(mm.results['fit_x']), 2))
        pp[:, 0] = mm.results['fit_x']
        pp[:, 1] = mm.results['fit_y']
        points.append(pp)

        # Image plot
        ax = fig.add_subplot(k)
        im = ax.imshow(mm.image, interpolation='None', aspect='equal',
                       cmap='cubehelix', vmin=0, vmax=700)
        ax.autoscale(False)
        ax.plot(mm.results['fit_y'] - 0.5, mm.results['fit_x'] - 0.5,
                'rx', mew=2, ms=5)
        ax.set_adjustable('box-forced')

        for i in np.arange(len(mm.results['fit_y'])):
            ax.annotate(str(i), xy=(mm.results['fit_y'][i] - 0.1,
                                    mm.results['fit_x'][i] - 0.1))

        k += 1

    # superposition of channels plot
    ax = fig.add_subplot(k)
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
    order = input('Reorden de points[1]: (ej: 0-1-4-3-2) ')
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

    for k in [0, 1]:
        # Image plot
        ax = fig.add_subplot(311 + k)
        ax.imshow(images[k], interpolation='None', aspect='equal',
                  cmap='cubehelix', vmin=0, vmax=700)
        ax.autoscale(False)
        ax.plot(points[k][:, 1] - 0.5, points[k][:, 0] - 0.5,
                'rx', mew=2, ms=5)
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
    ax.plot(points[0][:, 1] - 0.5, points[0][:, 0] - 0.5, 'rx', mew=2, ms=5)
    ax.plot(points[1][:, 1] - 0.5, points[1][:, 0] - 0.5, 'bs', mew=1, ms=5,
            markerfacecolor='None')
    ax.set_aspect('equal')
    ax.set_xlim(0, 266)
    ax.set_ylim(128, 0)


def remove_bad_points(points):

    ch = 0 if len(points[0]) > len(points[1]) else 1
    print('Number of registration points mismatch')
    print('Removing points from channel {} (0-{})'.format(ch, len(points[ch])))
    bpoints = input('Bad registration points: (ej: 1-7) ')
    bpoints = list(map(int, [l for l in bpoints.split('-')]))
    points[ch] = np.delete(points[ch], bpoints, 0)

    return points


def transformation_check(images, H, alpha):
    images2 = np.zeros((2, 128, 266), dtype=np.uint16)
    images2[0] = images[0]
    images2[1] = h_affine_transform(images[1], H)

    points = points_registration(images2)

    it = np.arange(len(points[0]))
    dist = np.array([np.linalg.norm(points[0][i] - points[1][i]) for i in it])
    print(dist)
    print('Mean distance: ', np.mean(dist))
    print('Maximum distance: ', np.max(dist))


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


def matrix_from_stack(filename=None, Hfilename=None):

            root = Tk()
            root.withdraw()
            folder = self.recWidget.filenameEdit.text()
            types = [('hdf5 files', '.hdf5'), ('tiff files', '.tiff'),
                     ('all files', '.*')]
            filename = filedialog.askopenfilename(filetypes=types, parent=root,
                                                  initialdir=folder,
                                                  title='Load bead stack')
            folder = os.path.split(filename)[0]
            arrayType = [('numpy array', '.npy')]
            Hname = filedialog.asksaveasfilename(filetypes=arrayType,
                                                 parent=root,
                                                 initialdir=folder,
                                                 title='Save affine matrix')
            Hname = Hname + '.npy'
            root.destroy()
            reg.matrix_from_stack(filename, Hname)

    images = load_hdf(filename)
    points = points_registration(images)
    H = matrix_from_points(points[0], points[1])
    print('Transformation matrix 1 --> 0')
    print(H)
    np.save(Hfilename, H)

if __name__ == '__main__':

    path = r'C:\Users\mdborde\Desktop\20160303 cruzi 568+647'
#    path = r'/home/federico/Desktop/data/'
    filename = 'tetraspeck_1.hdf5'
    Hfilename = os.path.join(path, 'Htransformation')

    matrix_from_stack(os.path.join(path, filename), Hfilename)

    print('Transformation checking')
    filename1 = 'tetraspeck.hdf5'
    images = load_hdf(os.path.join(path, filename1))
    transformation_check(images, np.load(Hfilename + '.npy'), 2)
