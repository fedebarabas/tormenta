# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: Federico Barabas
"""

import numpy as np
import h5py as hdf
import multiprocessing as mp

import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter

from tkinter import Tk, filedialog

import tormenta.analysis.tools as tools
import tormenta.analysis.maxima as maxima
from tormenta.tools import insertSuffix


def convert(word):
    splitted = word.split(' ')
    return splitted[0] + ''.join(x.capitalize() for x in splitted[1:])


def ask_files(title):
    root = Tk()
    filename = filedialog.askopenfilenames(parent=root, title=title)
    root.destroy()
    return filename


def ask_file(title):
    root = Tk()
    filename = filedialog.askopenfilename(parent=root, title=title)
    root.destroy()
    return filename


def split_two_colors(files):

    for name in files:

        with hdf.File(name, 'r') as ff:
            print(name)

            center = int(0.5*ff['data'].value.shape[1])

            with hdf.File(insertSuffix(name, '_ch0'), 'w') as ff0:
                ff0['data'] = ff['data'][:, center - 5 - 128:center - 5, :]

            with hdf.File(insertSuffix(name, '_ch1'), 'w') as ff1:
                ff1['data'] = ff['data'][:, center + 5:center + 5 + 128, :]


class Stack(object):
    """Measurement stored in a hdf5 file"""

    def __init__(self, filename=None, imagename='data'):

        if filename is None:
            filename = ask_file('Select hdf5 file')

        self.file = hdf.File(filename, 'r')

        # Loading of measurements (i.e., images) in HDF5 file
        self.imageData = self.file[imagename].value
        self.nframes = len(self.imageData)

        # Attributes loading as attributes of the stack
        self.attrs = self.file[imagename].attrs
        try:
            self.lambda_em = self.attrs['lambda_em']
        except:
            self.lambda_em = 670

        try:
            self.NA = self.attrs['NA']
        except:
            self.NA = 1.42

        try:
            self.nm_per_px = 1000 * self.attrs['element_size_um'][2]
        except:
            self.nm_per_px = 120

        self.frame = 0
        self.fwhm = tools.get_fwhm(self.lambda_em, self.NA) / self.nm_per_px
        self.win_size = int(np.ceil(self.fwhm))

        self.kernel = tools.kernel(self.fwhm)
        self.xkernel = tools.xkernel(self.fwhm)

    def localize_molecules(self, ran=(0, None), fit_model='2d'):

        if ran[1] is None:
            ran = (0, self.nframes)

        self.fit_parameters = maxima.fit_par(fit_model)
        self.dt = maxima.results_dt(self.fit_parameters)

        cpus = mp.cpu_count()
        step = (ran[1] - ran[0]) // cpus
        chunks = [[i*step, (i + 1)*step] for i in np.arange(cpus)]
        chunks[-1][1] = ran[1]

        max_args = (self.fit_parameters, self.dt, self.fwhm, self.win_size,
                    self.kernel, self.xkernel)
        args = [[self.imageData[i:j], i, fit_model, max_args]
                for i, j in chunks]

        pool = mp.Pool(processes=cpus)
        results = pool.map(localize_chunk, args)
        pool.close()
        pool.join()
        self.molecules = np.concatenate(results[:])

    def scatter_plot(self):
        plt.plot(self.molecules['fit_y'], self.molecules['fit_x'], 'bo',
                 markersize=0.2)
        plt.xlim(0, self.imageData[0].shape[0])
        plt.ylim(0, self.imageData[0].shape[1])

    def filter_results(self, trail=True):

        # TODO: filter by parameters
        if trail:

            # Find trails: local maxima in the same pixel in consecutive frames
            sorted_m = np.sort(self.molecules, order=['maxima', 'frame'])

            i = 1
            cuts = []
            while i < len(sorted_m):

                try:

                    while tools.overlaps(sorted_m['maxima'][i - 1],
                                         sorted_m['maxima'][i], 2):
                        i = i + 1

                    cuts.append(i)
                    i = i + 1

                except:
                    pass

#            sorted_m = np.array_split(sorted_m, cuts)

    def __exit__(self):
        self.file.close()

    def close(self):
        self.file.close()


def bkg_estimation(data_stack, window=100):
    ''' Background estimation. It's a running (time) mean.
    Hoogendoorn et al. in "The fidelity of stochastic single-molecule
    super-resolution reconstructions critically depends upon robust background
    estimation" recommend a median filter, but that takes too long, so we're
    using an uniform filter.'''

    intensity = np.mean(data_stack, (1, 2))
    norm_data = data_stack / intensity[:, np.newaxis, np.newaxis]
#    bkg_estimate = median_filter(norm_data, size=(window, 1, 1))
    bkg_estimate = uniform_filter(norm_data, size=(window, 1, 1))
    bkg_estimate *= intensity[:, np.newaxis, np.newaxis]

    return bkg_estimate


def localize_chunk(args, index=0):

    stack, init_frame, fit_model, max_args = args
    fit_parameters, res_dt, fwhm, win_size, kernel, xkernel = max_args
    n_frames = len(stack)

    bkg_stack = bkg_estimation(stack)

    # I create a big array, I'll keep the non-null part at the end
    nn = int(np.ceil((n_frames + 1)*np.prod(stack.shape[1:])/(win_size + 1)))
    results = np.zeros(nn, dtype=res_dt)

#    mol_per_frame = np.zeros(n_frames,
#                             dtype=[('frame', int), ('molecules', int)])
#    frame = init_frame

    for n in np.arange(n_frames):

        # fit all molecules in each frame
        maxi = maxima.Maxima(stack[n], fit_parameters, res_dt, fwhm, win_size,
                             kernel, xkernel, bkg_stack[n])
        maxi.find()

        maxi.getParameters()
        maxi.fit(fit_model)

        # save frame number and fit results
        results[index:index + len(maxi.results)] = maxi.results
        results['frame'][index:index + len(maxi.results)] = init_frame + n

        # save number of molecules per frame
#            mol_per_frame['frame'][frame - init] = frame
#            mol_per_frame['molecules'][frame - init] = len(maxi.results)

        index += len(maxi.results)


#        progress = np.round((100 * (frame - init) / len(frames)), 2)
#        print('{}% done'.format(progress), end="\r")

    return results[0:index]

if __name__ == "__main__":

#    import matplotlib.pyplot as plt
#    se = Stack(r'/home/federico/Desktop/20160212 Tetraspeck registration/filename_9.hdf5')
#    mm = maxima.Maxima(se.imageData[10], se.fwhm)
#    mm.find()
#    peak = mm.area(mm.image, 5)
#    plt.imshow(peak, interpolation='None')
#    gme = maxima.fit_GME(peak, mm.fwhm)
#    mle = maxima.fit_area(peak, mm.fwhm, np.min(peak))
#    print(mle)
#    print(gme)
#
#    plt.plot(mle[2] - 0.5, mle[1] - 0.5, 'rx', mew=2, ms=5)
#    plt.plot(gme[0] - 0.5, gme[1] - 0.5, 'bs', mew=1, ms=5, markerfacecolor='none')
#    plt.colorbar()
#    plt.show()

    split_two_colors(ask_files('Select hdf5 file'))

#    stack = Stack()
#    maxima = Peaks(stack.image[10], stack.fwhm)
#    maxima.find(stack.kernel, stack.xkernel)
#    plt.imshow(maxima.image, interpolation='nearest')
#    plt.colorbar()
#    plt.plot(maxima.positions[:, 1], maxima.positions[:, 0],
#             'ro', markersize=10, alpha=0.5)
#
#    maxima.fit()
#    print(maxima.results)

#    stack.localize_molecules(init=10, end=20)
