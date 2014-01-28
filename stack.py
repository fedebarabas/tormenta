# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: federico
"""

import numpy as np
from scipy.ndimage.filters import convolve
import h5py as hdf

from airygauss import fwhm


def gauss(x, center, fwhm):
    return np.exp(- 4 * np.log(2) * (x - center)**2 / fwhm**2)


def kernel(fwhm):
    window = np.ceil(fwhm) + 3
    x = np.arange(0, window)
    y = x
    xx, yy = np.meshgrid(x, y, sparse=True)
    matrix = gauss(xx, x.mean(), fwhm) * gauss(yy, y.mean(), fwhm)
    matrix = matrix - matrix.sum() / matrix.size
    return matrix


def xkernel(fwhm):
    window = np.ceil(fwhm) + 3
    x = np.arange(0, window)
    matrix = gauss(x, x.mean(), fwhm)
    matrix = matrix - matrix.sum() / matrix.size
    return matrix


def drop_overlapping(peaks, size):
    """We exclude from the analysis all the peaks that have their fitting
    windows overlapped. The size parameter is the number of pixels from the
    local maxima to the edge of this window.
    """

    no_overlaps = np.zeros(peaks.shape, dtype=int)

    def does_not_overlap(p1, p2):
        return max(abs(p1[1] - p2[1]), abs(p1[0] - p2[0])) > 2*size

    nov_peaks = 0
    for i in np.arange(len(peaks)):

        if all(map(lambda x: does_not_overlap(peaks[i], x),
                   np.delete(peaks, i, 0))):

            no_overlaps[nov_peaks] = peaks[i]
            nov_peaks += 1

    return no_overlaps[:nov_peaks]


class Peaks(object):

    def find(self, image, kernel, xkernel, alpha=3, size=2):
        """Peak finding routine.
        Alpha is the amount of standard deviations used as a threshold of the
        local maxima search. Size is the semiwidth of the fitting window.
        Adapted from http://stackoverflow.com/questions/16842823/
                            peak-detection-in-a-noisy-2d-array

        Keep in mind the 'border issue': some peaks, if they are at a distance
        equal to 'size' from the border of the image, won't be centered in
        the maximum value.
        """
        # Image cropping to avoid border problems
        image_crop = image[size:-size, size:-size]
        shape = image_crop.shape

        # Noise removal by convolving with a null sum gaussian. Its FWHM
        # has to match the one of the objects we want to detect.
        image_conv = convolve(image_crop.astype(float), kernel)

#        image_temp = deepcopy(image_conv)
        image_mask = np.zeros(shape, dtype=bool)

        std = image_conv.std()
        peaks = np.zeros((np.ceil(image_crop.size / (2*size + 1)**2), 2),
                         dtype=int)
        peak_ct = 0

        while 1:

            k = np.argmax(np.ma.masked_array(image_conv, image_mask))

            # index juggling
            j, i = np.unravel_index(k, shape)
            if(image_conv[j, i] >= alpha*std):

                # Saving the peak relative to the original image
                peaks[peak_ct] = [j + size, i + size]

                # this is the part that masks already-found peaks
                x = np.arange(i - size, i + size + 1)
                y = np.arange(j - size, j + size + 1)
                xv, yv = np.meshgrid(x, y)
                # the clip handles cases where the peak is near the image edge
                image_mask[yv.clip(0, shape[0] - 1),
                           xv.clip(0, shape[1] - 1)] = True

                peak_ct += 1

            else:
                break

        self.backgrd_est = np.ma.masked_array(image_crop, image_mask).mean()

        # Drop overlapping
        peaks = drop_overlapping(peaks[:peak_ct], size)
        self.positions = peaks

        # Peak parameters
        roundness = np.zeros(len(peaks))
        brightness = np.zeros(len(peaks))

        sharpness = np.zeros(len(peaks))
        mask = np.zeros((2*size + 1, 2*size + 1), dtype=bool)
        mask[size, size] = True

        for i in np.arange(len(peaks)):
            # tuples make indexing easier (see below)
            p = tuple(peaks[i])
            p_crop = tuple(peaks[i] - size)

            # Sharpness
            masked = np.ma.masked_array(peak(image, p, size), mask)
            sharpness[i] = image[p] / (image_conv[p_crop] * masked.mean())

            # Roundness
            hx = np.dot(peak(image, p, size)[2, :], xkernel)
            hy = np.dot(peak(image, p, size)[:, 2], xkernel)
            roundness[i] = 2 * (hy - hx) / (hy + hx)

            # Brightness
            brightness[i] = 2.5 * np.log(image_conv[p_crop] / alpha*std)

        self.size = size
        self.alpha = alpha

        self.sharpness = sharpness
        self.roundness = roundness
        self.brightness = brightness


def peak(img, p, size):
    """Caller for the area around the peak."""

    return img[p[0] - size:p[0] + size + 1, p[1] - size:p[1] + size + 1]


class Stack(object):
    """Measurement stored in a hdf5 file"""

    def __init__(self, filename=None, imagename='frames'):

        if filename is None:

            import tkFileDialog as filedialog
            from Tkinter import Tk

            root = Tk()
            filename = filedialog.askopenfilename(parent=root,
                                                  title='Select hdf5 file')
            root.destroy()

        hdffile = hdf.File(filename, 'r')

        # Loading of measurements (i.e., images) in HDF5 file
        for measure in hdffile.items():
            setattr(self, measure[0], measure[1])

        # Attributes loading as attributes of the stack
        for att in hdffile.attrs.items():
            setattr(self, att[0], att[1])

        self.frame = 0
        self.fwhm = fwhm(self.lambda_em, self.NA) / self.nm_per_px
        self.kernel = kernel(self.fwhm)
        self.xkernel = xkernel(self.fwhm)


if __name__ == "__main__":

#    import matplotlib.pyplot as plt

    stack = Stack()
    peaks = Peaks()
    peaks.find(stack.image[10], stack.kernel, stack.xkernel)
#    plt.imshow(stack.image[10], interpolation='nearest')
#    plt.colorbar()
#    plt.plot(peaks.positions[:, 1], peaks.positions[:, 0],
#             'ro', markersize=10, alpha=0.5)
