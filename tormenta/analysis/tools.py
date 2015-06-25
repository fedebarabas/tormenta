# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:01 2014

@author: fbaraba
"""


def mode(array):

    hist, bin_edges = np.histogram(array, bins=array.max() - array.min())
    hist_max = hist.argmax()
    return (bin_edges[hist_max + 1] + bin_edges[hist_max]) / 2


def overlaps(p1, p2, minD):
    return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) <= minD


def dropOverlapping(maxima, minD):
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
            nov_maxima += 1

    return noOverlaps[:n]


def gauss(x, x0, fwhm):
    """ Returns the closest gaussian function to an Airy disk centered in x0
    and a full width half maximum equal to fwhm."""
    return np.exp(- 4 * np.log(2) * (x - x0)**2 / fwhm**2)


def kernel(fwhm):
    """ Returns the kernel of a convolution used for finding objects of a
    full width half maximum fwhm in an image."""
    window = np.ceil(fwhm) + 3
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
