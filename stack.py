# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:44:59 2013

@author: federico
"""

import numpy as np
import h5py as hdf


def denoise(frame):
    """Noise removal by substracting most common signal in the frame"""

    histo = np.histogram(frame, bins=100)

    # Histogram smoothing
    histo_s = np.convolve(histo[0], np.ones(5) / 5)

    noise = histo[1][np.where(histo_s == np.max(histo_s))]

    return frame - noise


class Stack(object):
    """Measurement stored in a hdf5 file"""

    def __init__(self, filename, imagename='frames'):

        hdffile = hdf.File(filename, 'r')

        self.image = hdffile[imagename]
        self.size = hdffile[imagename].shape[1:3]
        self.nframes = hdffile[imagename].shape[0]
        self.frame = 0
