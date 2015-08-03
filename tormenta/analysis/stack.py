# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: federico
"""

import numpy as np
import h5py as hdf

import tormenta.analysis.tools as tools

# data-type definitions
parameters_2d = [('amplitude', float), ('x0', float), ('y0', float),
                 ('background', float)]
parameters = [('frame', int), ('maxima', np.int, (2,)), ('photons', float),
              ('sharpness', float), ('roundness', float),
              ('brightness', float)]
dt_2d = np.dtype(parameters + parameters_2d)


def convert(word):
    splitted = word.split(' ')
    return splitted[0] + ''.join(x.capitalize() for x in splitted[1:])


class Stack(object):
    """Measurement stored in a hdf5 file"""

    def __init__(self, filename=None, imagename='data'):

        if filename is None:

            import tkFileDialog as filedialog
            from Tkinter import Tk

            root = Tk()
            filename = filedialog.askopenfilename(parent=root,
                                                  title='Select hdf5 file')
            root.destroy()

        self.file = hdf.File(filename, 'r')

        # Loading of measurements (i.e., images) in HDF5 file
        self.imageData = self.file[imagename].value

        # Attributes loading as attributes of the stack
        self.attrs = self.file[imagename].attrs
#        try:
#            self.attrs['lambda_em']
#        except:
#        self.attrs['lambda_em'] = 670
        self.lambda_em = 670

#        try:
#            self.attrs['NA']
#        except:
        self.NA = 1.42

        try:
            self.nm_per_px = 1000 * self.attrs['element_size_um'][2]
        except:
            self.nm_per_px = 120

        self.frame = 0
        self.fwhm = tools.fwhm(self.lambda_em, self.NA) / self.nm_per_px
        self.win_size = np.ceil(self.fwhm)

        self.kernel = tools.kernel(self.fwhm)
        self.xkernel = tools.xkernel(self.fwhm)

    def __exit__(self):
        self.file.close()

    def localize_molecules(self, init=0, end=None, fit_model='2d'):

        if end is None:
            end = self.nframes

        # I create a big array, I'll keep the non-null part at the end
        # frame | peaks.results
        self.frames = np.arange(init, end + 1)

        if fit_model is '2d':
            self.dt = dt_2d

        results = np.zeros(len(self.frames)*np.prod(self.size)
                           / (self.win_size + 1), dtype=self.dt)

        mol_per_frame = np.zeros(len(self.frames),
                                 dtype=[('frame', int), ('molecules', int)])
        index = 0

        for frame in self.frames:

            # fit all molecules in each frame
            maxi = maxima.Maxima(stack.image[frame], stack.fwhm)
            maxi.find(stack.kernel, stack.xkernel)
            maxi.fit(fit_model)

            # save frame number and fit results
            results[index:index + len(maxi.results)] = maxi.results
            results['frame'][index:index + len(maxi.results)] = frame

            # save number of molecules per frame
            mol_per_frame['frame'][frame - init] = frame
            mol_per_frame['molecules'][frame - init] = len(maxi.results)

            index = index + len(maxi.results)

            print(100 * (frame - init) / len(self.frames), '% done')

        # final results table
        self.molecules = results[0:index]

    def filter_results(self, trail=True):

        ### TODO: filter by parameters

        if trail:

            # Find trails: local maxima in the same pixel in consecutive frames
            sorted_m = np.sort(self.molecules, order=['maxima', 'frame'])

            i = 1
            cuts = []
            while i < len(sorted_m):

                try:

                    while does_overlap(sorted_m['maxima'][i - 1],
                                       sorted_m['maxima'][i], 2):
                        i = i + 1

                    cuts.append(i)
                    i = i + 1

                except:
                    pass

#            sorted_m = np.array_split(sorted_m, cuts)

    def close(self):
        self.file.close()


if __name__ == "__main__":

#    import matplotlib.pyplot as plt

    stack = Stack()
#    maxima = Peaks(stack.image[10], stack.fwhm)
#    maxima.find(stack.kernel, stack.xkernel)
#    plt.imshow(maxima.image, interpolation='nearest')
#    plt.colorbar()
#    plt.plot(maxima.positions[:, 1], maxima.positions[:, 0],
#             'ro', markersize=10, alpha=0.5)
#
#    image = stack.image[10]
#    pico = maxima.get_peak(10)
#
#    maxima.fit()
#    print(maxima.results)

    stack.localize_molecules(init=10, end=20)
