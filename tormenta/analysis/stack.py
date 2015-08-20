# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: federico
"""

import numpy as np
import h5py as hdf
import multiprocessing as mp

import tormenta.analysis.tools as tools
import tormenta.analysis.maxima as maxima


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
        chunks = [[i*step, (i + 1)*step - 1] for i in np.arange(cpus)]
        chunks[-1][1] = ran[1] - 1

        max_args = (self.fit_parameters, self.dt, self.fwhm, self.win_size,
                    self.kernel, self.xkernel)
        args = [[self.imageData[i:j], i, fit_model, max_args]
                for i, j in chunks]

        pool = mp.Pool(processes=cpus)
        results = pool.map(localize_chunk, args)
        self.molecules = np.concatenate(results[:])

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


def localize_chunk(args):

    stack, init_frame, fit_model, max_args = args
    fit_parameters, res_dt, fwhm, win_size, kernel, xkernel = max_args
    n_frames = len(stack)

    # I create a big array, I'll keep the non-null part at the end
    results = np.zeros((n_frames + 1)*np.prod(stack.shape[1:])//(win_size + 1),
                       dtype=res_dt)

#    mol_per_frame = np.zeros(n_frames,
#                             dtype=[('frame', int), ('molecules', int)])
    index = 0
    frame = init_frame

    for n in np.arange(n_frames):

        frame += 1

        # fit all molecules in each frame
        maxi = maxima.Maxima(stack[n], fit_parameters, res_dt, fwhm, win_size,
                             kernel, xkernel)
        maxi.find()

        try:
            maxi.getParameters()
            maxi.fit(fit_model)

            # save frame number and fit results
            results[index:index + len(maxi.results)] = maxi.results
            results['frame'][index:index + len(maxi.results)] = frame

            # save number of molecules per frame
#            mol_per_frame['frame'][frame - init] = frame
#            mol_per_frame['molecules'][frame - init] = len(maxi.results)

            index += len(maxi.results)
        except IndexError:
            pass

#        progress = np.round((100 * (frame - init) / len(frames)), 2)
#        print('{}% done'.format(progress), end="\r")

    return results[0:index]

#if __name__ == "__main__":

#    import matplotlib.pyplot as plt

#    stack = Stack()
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

#    stack.localize_molecules(init=10, end=20)
