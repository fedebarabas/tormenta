# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:01 2014

@author: fbaraba
"""
import numpy as np


def Haffine_from_points(fp, tp):
    """ find H, affine transformation, such that tp is affine transf of fp.
    Taken from http://www.janeriksolem.net/2009/06/
    affine-transformations-and-warping.html
    """

    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')

    # Condition points
    # -from points-
    m = np.mean(fp[:2], axis=1)
    maxstd = max(np.std(fp[:2], axis=1))
    C1 = np.diag([1/maxstd, 1/maxstd, 1])
    C1[0][2] = -m[0]/maxstd
    C1[1][2] = -m[1]/maxstd
    fp_cond = np.dot(C1, fp)

    # -to points-
    m = np.mean(tp[:2], axis=1)
    C2 = C1.copy()  # must use same scaling for both point sets
    C2[0][2] = -m[0]/maxstd
    C2[1][2] = -m[1]/maxstd
    tp_cond = np.dot(C2, tp)

    # conditioned points have mean zero, so translation is zero
    A = np.concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = np.linalg.svd(A.T)

    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]

    tmp2 = np.concatenate((np.dot(C, np.linalg.pinv(B)),
                           np.zeros((2, 1))), axis=1)
    H = np.vstack((tmp2, [0, 0, 1]))

    # decondition
    H = np.dot(np.linalg.inv(C2), np.dot(H, C1))

    return H / H[2][2]

#transformed_im = ndimage.affine_transform(im,A,b,size)
#
#transforms the image patch im with A a linear transformation and b a translation vector as above. The optional argument size can be used to specify the size of the output image. The default is an image with the same size as the original. To see how this works, try running the following commands:
#
#>>>from PIL import Image
#>>>from scipy import ndimage
#>>>from pylab import *
#
#>>>im = array(Image.open('empire.jpg').convert('L'))
#>>>H = array([[1.4,0.05,-100],[0.05,1.5,-100],[0,0,1]])
#>>>im2 = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]))
#
#>>>figure()
#>>>gray()
#>>>imshow(im2)
#>>>show()
#https://stackoverflow.com/questions/27546081/determining-a-homogeneous-affine-transformation-matrix-from-six-points-in-3d-usi/27547597#27547597
#http://elonen.iki.fi/code/misc-notes/affine-fit/

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
        overlapFunction = lambda x: not(overlaps(maxima[i], x, d))
        overlapsList = map(overlapFunction, np.delete(maxima, i, 0))
        if all(overlapsList):
            noOverlaps[n] = maxima[i]
            n += 1

    return noOverlaps[:n]


def gauss(x, x0, fwhm):
    """ Returns the closest gaussian function to an Airy disk centered in x0
    and a full width half maximum equal to fwhm."""
    return np.exp(- 4 * np.log(2) * (x - x0)**2 / fwhm**2)


def kernel(fwhm):
    """ Returns the kernel of a convolution used for finding objects of a
    full width half maximum fwhm in an image."""
#    window = np.ceil(fwhm) + 3
    window = int(np.ceil(fwhm)) + 2
    x = np.arange(0, window)
    y = x
    xx, yy = np.meshgrid(x, y, sparse=True)
    matrix = gauss(xx, x.mean(), fwhm) * gauss(yy, y.mean(), fwhm)
    matrix /= matrix.sum()
    return matrix


def xkernel(fwhm):
    window = np.ceil(fwhm) + 3
    x = np.arange(0, window)
    matrix = gauss(x, x.mean(), fwhm)
    matrix = matrix - matrix.sum() / matrix.size
    return matrix
