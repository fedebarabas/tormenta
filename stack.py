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


def denoise(frame):
    """Noise removal by substracting most common signal in the frame"""

    histo = np.histogram(frame, bins=100)

    # Histogram smoothing
    histo_s = np.convolve(histo[0], np.ones(5) / 5)

    noise = histo[1][np.argmax(histo_s)]

    return frame - noise


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

if __name__ == "__main__":

    stack = Stack()
