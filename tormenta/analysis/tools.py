# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:03:01 2014

@author: Federico Barabas
"""

import numpy as np
from scipy.special import jn
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
# rc('text', usetex=True)


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
    NA = 1.42

#    x = np.arange(-1.5, 1.5, 0.01)
    x = np.arange(-600, 600)
    y = airy(NA * x / wavelength)
    fw = get_fwhm(wavelength, NA)
    fit = best_gauss(x, 0, fw)

    print('FWHM is', np.round(fw))

    plt.figure()
    plt.plot(x, y, label='Airy disk', lw=4)
    plt.plot(x, fit, label='Gaussian fit', lw=4)
    plt.xlabel('Position [nm]', fontsize='large')
    plt.ylabel('Intensity', fontsize='large')
    plt.legend()
    plt.grid('on')
    plt.show()


def mode(array):
    hist, bin_edges = np.histogram(array, bins=array.max() - array.min())
    hist_max = hist.argmax()
    return (bin_edges[hist_max + 1] + bin_edges[hist_max]) / 2


def overlaps(p1, p2, d):
    return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) <= d


def dropOverlapping(maxx, d):
    """We exclude from the analysis all the maxima in maxx that have their
    fitting windows overlapped, i.e., the distance between them is less than
    'd'."""

    noOverlaps = np.zeros(maxx.shape, dtype=int)  # Final array

    n = 0
    for i in np.arange(len(maxx)):
        def overlapFunction(x):
            return not(overlaps(maxx[i], x, d))
        overlapsList = map(overlapFunction, np.delete(maxx, i, 0))
        if all(overlapsList):
            noOverlaps[n] = maxx[i]
            n += 1

    return noOverlaps[:n]


def kernel(fwhm):
    """ Returns the kernel of a convolution used for finding objects of a
    full width half maximum fwhm (in pixels) in an image."""
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
