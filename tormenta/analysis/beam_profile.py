# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:19:37 2014
Created on Wed May 13 23:35:23 2015

@author: federico
"""

import numpy as np
from scipy.signal import argrelextrema
from tkinter import Tk, filedialog

from stack import Stack


def loadStacks(ask, folder=None):
    # Get filenames from user
    try:
        root = Tk()
        stacksNames = filedialog.askopenfilenames(parent=root, title=ask,
                                                  initialdir=folder)
        root.destroy()
    except OSError:
        print("No files selected!")

    # Fix for names with whitespace.
    # Taken from: http://stackoverflow.com/questions/9227859/
    # tkfiledialog-not-converting-results-to-a-python-list-on-windows
    folder = os.path.split(stacksNames[0])[0]
    if isinstance(stacksNames, list) or isinstance(stacksNames, tuple):
        return stacksNames, folder
    else:
        return stacksNames.strip('{}').split('} {'), folder


def beamProfile(ask, folder=None, shape=(512, 512)):

    stacks, folder = loadStacks(ask, folder)

    n = len(stacks)
    profile = np.zeros(shape)
    norm = 0

    for filename in stacks:
        print(filename)
        stack = Stack(filename=filename)
        meanFrame = stack.imageData.mean(0)
        meanInt = meanFrame.mean()
        profile += meanFrame / meanInt
        norm += meanInt
        stack.close()

    norm /= n

    hist, edg = np.histogram(profile, bins=100)
    thres = edg[argrelextrema(hist, np.less)[0][0] + 1]
    beam_mask = np.zeros(shape=profile.shape)
    beam_mask[profile < thres] = True
    beamProfile = np.ma.masked_array(profile, beam_mask)

    return beamProfile, norm, folder


def frame(image, center=(256, 256), shape=(128, 128)):

    return image[center[0] - int(shape[0] / 2):center[0] + int(shape[0] / 2),
                 center[1] - int(shape[1] / 2):center[1] + int(shape[1] / 2)]

# def analyze_beam(epinames=None, tirfnames=None):
#
#    if epinames is None:
#        epinames = load_files('epi')
#        tirfnames = load_files('tirf')
#
#    epi_mean = beam_mean(epinames)
#    tirf_mean = beam_mean(tirfnames)
#
#    tirf_factor = frame(tirf_mean).mean() / frame(epi_mean).mean()
#    frame_factor = frame(tirf_mean).mean() / tirf_mean.mean()
#    variance = 100 * frame(tirf_mean).std() / frame(tirf_mean).mean()
#
#    return tirf_factor, frame_factor, variance
#
if __name__ == "__main__":

    from PIL import Image
    import matplotlib.pyplot as plt

    profileEPI, normEPI, folder = beamProfile('epi')
    profileTIRF, normTIRF, folder = beamProfile('tirf', folder)
    TIRFactor = normTIRF / normEPI

    TIRFrameFactor = frame(profileTIRF).mean() / profileTIRF.mean()
    EPIFrameFactor = frame(profileEPI).mean() / profileEPI.mean()
    TIRstd = 100 * frame(profileTIRF).std() / frame(profileTIRF).mean()
    EPIstd = 100 * frame(profileEPI).std() / frame(profileEPI).mean()

    print('TIRF intensity factor', TIRFactor)
    print('EPI Frame factor', EPIFrameFactor)
    print('TIRF Frame factor', TIRFrameFactor)
    print('%std for EPI frame', EPIstd)
    print('%std for TIRF frame', TIRstd)

    im = Image.fromarray(profileEPI)
    im.save('profileEPI.tiff')
    im = Image.fromarray(profileTIRF)
    im.save('profileTIRF.tiff')

#    f = plt.figure()
    plt.subplot(2,  1, 1)
    plt.imshow(profileEPI, interpolation='None', cmap=cm.cubehelix)
    plt.title('EPI')
    plt.colorbar()
    plt.text(700, 100, 'TIRF intensity factor={}'.format(np.round(TIRFactor, 2)))

    plt.subplot(2, 1, 2)
    plt.imshow(profileTIRF, interpolation='None', cmap=cm.cubehelix)
    plt.title('TIRF')
    plt.colorbar()

    plt.show()
