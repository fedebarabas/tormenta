# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:19:37 2014
Created on Wed May 13 23:35:23 2015

@author: federico
"""

import numpy as np
from scipy.signal import argrelextrema
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt

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
    # Untested for not centered frames

    return image[center[0] - int(shape[0] / 2):center[0] + int(shape[0] / 2),
                 center[1] - int(shape[1] / 2):center[1] + int(shape[1] / 2)]


def analyzeBeam():
    """
    Script for loading EPI and TIRF images of homogeneous samples for
    measuring the illumination beam profile.
    It loads as many images as you have for both illumination schemes and
    then it calculates the means and correction factors.
    """

    profileEPI, normEPI, folder = beamProfile('Select EPI profiles')
    profileTIRF, normTIRF, folder = beamProfile('Select TIRF profiles', folder)
    TIRFactor = normTIRF / normEPI

    TIRFrameFactor = frame(profileTIRF).mean() / profileTIRF.mean()
    EPIFrameFactor = frame(profileEPI).mean() / profileEPI.mean()
    TIRstd = 100 * frame(profileTIRF).std() / frame(profileTIRF).mean()
    EPIstd = 100 * frame(profileEPI).std() / frame(profileEPI).mean()

    im = Image.fromarray(profileEPI)
    im.save(os.path.join(folder, 'profileEPI.tiff'))
    im = Image.fromarray(profileTIRF)
    im.save(os.path.join(folder, 'profileTIRF.tiff'))

    # EPI profile
    plt.subplot(2,  2, 1)
    plt.imshow(profileEPI, interpolation='None', cmap=cm.cubehelix)
    plt.title('EPI profile')
    plt.colorbar()
    plt.text(800, 100,
             'EPI frame factor={}'.format(np.round(EPIFrameFactor, 2)))
    plt.text(800, 150,
             'EPI % standard dev={}'.format(np.round(EPIstd, 2)))

    # TIRF profile
    plt.subplot(2, 2, 3)
    plt.imshow(profileTIRF, interpolation='None', cmap=cm.cubehelix)
    plt.title('TIRF profile')
    plt.colorbar()
    plt.text(800, 100,
             'TIRF frame factor={}'.format(np.round(TIRFrameFactor, 2)))
    plt.text(800, 150,
             'TIRF % standard dev={}'.format(np.round(TIRstd, 2)))
    plt.text(800, 250,
             'TIRF intensity factor={}'.format(np.round(TIRFactor, 2)))

    plt.show()

if __name__ == "__main__":

    analyzeBeam()
