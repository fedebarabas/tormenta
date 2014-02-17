# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: federico
"""

import numpy as np

import h5py as hdf

from airygauss import fwhm
from peaks import Peaks

# data-type definitions
parameters_2d = [('amplitude', float), ('x0', float), ('y0', float),
                 ('background', float)]
parameters = [('frame', int), ('photons', float), ('sharpness', float),
              ('roundness', float), ('brightness', float)]
dtype_2d = np.dtype(parameters + parameters_2d)


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


def get_mode(array):

    hist, bin_edges = np.histogram(array, bins=array.max() - array.min())
    hist_max = hist.argmax()
    return (bin_edges[hist_max + 1] + bin_edges[hist_max]) / 2


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
        self.win_size = np.ceil(self.fwhm)

        self.kernel = kernel(self.fwhm)
        self.xkernel = xkernel(self.fwhm)

    def find_molecules(self, init=0, end=None, fit_model='2d'):

        if end is None:
            end = self.nframes

        # I create a big array, I'll keep the non-null part at the end
        # frame | peaks.results
        nframes = end - init

        if fit_model is '2d':
            results = np.zeros(nframes*np.prod(self.size)/(self.win_size + 1),
                               dtype=dtype_2d)

        mol_per_frame = np.recarray(nframes,
                                    dtype=[('frame', int), ('molecules', int)])
        index = 0

        for frame in np.arange(init, end):

            # fit all molecules in each frame
            peaks = Peaks(stack.image[frame], stack.fwhm)
            peaks.find(stack.kernel, stack.xkernel)
            peaks.fit(fit_model)

            # save frame number and fit results
            results[index:index + len(peaks.results)] = peaks.results
            results['frame'][index:index + len(peaks.results)] = frame

            # save number of molecules per frame
            mol_per_frame['frame'][frame - init] = frame
            mol_per_frame['molecules'][frame - init] = len(peaks.results)

            index = index + len(peaks.results)

            print(100 * (frame - init) / nframes, '% done')

        # final results table
        self.molecules = results[0:index]

    def filter_results(self, join=True):

        self.molecules

if __name__ == "__main__":

#    import matplotlib.pyplot as plt

    stack = Stack()
#    peaks = Peaks(stack.image[10], stack.fwhm)
#    peaks.find(stack.kernel, stack.xkernel)
#    plt.imshow(peaks.image, interpolation='nearest')
#    plt.colorbar()
#    plt.plot(peaks.positions[:, 1], peaks.positions[:, 0],
#             'ro', markersize=10, alpha=0.5)
#
##    image = stack.image[10]
#    pico = peaks.get_peak(10)
#
#    peaks.fit()
#    print(peaks.results)

    stack.find_molecules(init=10, end=20)
