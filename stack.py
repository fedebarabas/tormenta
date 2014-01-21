# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: federico
"""

import numpy as np
from scipy.ndimage.filters import convolve
import h5py as hdf
from copy import deepcopy

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


def denoise(frame, kernel):
    """Noise removal by convolving with a null sum gaussian.
    Its FWHM matches the one of the objects we want to detect."""
    return convolve(frame.astype(float), kernel)


def get_max(image, alpha=5, size=10):
    i_out = []
    j_out = []
    image_temp = deepcopy(image)
    while True:
        k = np.argmax(image_temp)
        j, i = np.unravel_index(k, image_temp.shape)
        if(image_temp[j, i] >= alpha*image.std()):
            i_out.append(i)
            j_out.append(j)
            x = np.arange(i-size, i+size)
            y = np.arange(j-size, j+size)
            xv, yv = np.meshgrid(x, y)
            image_temp[yv.clip(0, image_temp.shape[0]-1),
                       xv.clip(0, image_temp.shape[1]-1)] = 0
#            print(xv)
        else:
            break

    return i_out, j_out


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
