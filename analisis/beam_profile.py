# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:19:37 2014
Created on Wed May 13 23:35:23 2015

@author: federico
"""

import numpy as np

t_on = 10
r_min = (15, 25)
l = 85

# seguir con esto
def load_stack(filename, shape=shapes, dtype=np.dtype('>u2')):

    # assuming big-endian
    data = np.memmap(filename, dtype=dtype, mode='r')
    return data.reshape(shape)


def mean_frame(stack, start_frame=0):
    '''Get the mean of all pixels from start_frame'''

    return stack[start_frame:].mean(0)


def get_beam(image):

    hist, edg = np.histogram(image, bins=50)

    # We get the inside of the beam as the pixels with intesity higher than the
    # minimum of the histogram of the image, between the background and the
    # beam distribution
    thres = edg[np.argmin(hist[:np.argmax(hist)])]
    beam_mask = np.zeros(shape=image.shape)
    beam_mask[image < thres] = True

    return np.ma.masked_array(image, beam_mask)


def beam_mean(filenames):

    mean_frames = np.zeros((len(filenames), shapes[1], shapes[2]))

    for i in np.arange(len(filenames)):
        print(filenames[i])
        data = load_stack(filenames[i])
        mean_frames[i] = mean_frame(data, t_on)

    return get_beam(mean_frames.mean(0))


def frame(image, r_min=r_min, l=l):

    return image[r_min[1]:r_min[1] + l, r_min[0]:r_min[0] + l]


def analyze_beam(epinames=None, tirfnames=None):

    if epinames is None:
        epinames = load_files('epi')
        tirfnames = load_files('tirf')

    epi_mean = beam_mean(epinames)
    tirf_mean = beam_mean(tirfnames)

    tirf_factor = frame(tirf_mean).mean() / frame(epi_mean).mean()
    frame_factor = frame(tirf_mean).mean() / tirf_mean.mean()
    variance = 100 * frame(tirf_mean).std() / frame(tirf_mean).mean()

    return tirf_factor, frame_factor, variance

if __name__ == "__main__":

#    %load_ext autoreload
#    %autoreload 2

    import sys

    repos = 'P:\\Private\\repos'
    sys.path.append(repos)

    import switching_analysis.beam_profile as bp

    epi_fov = bp.beam_mean(bp.load_files('epi'))
    tirf_fov = bp.beam_mean(bp.load_files('tirf'))

    tirf_factor = frame(tirf_fov).mean() / frame(epi_fov).mean()
    frame_factor = frame(tirf_fov).mean() / tirf_fov.mean()
    std = 100 * frame(tirf_fov).std() / frame(tirf_fov).mean()

#   plt.imshow(tirf_mean, interpolation='none')
#   plt.colorbar()