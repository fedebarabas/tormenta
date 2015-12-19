# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:19:37 2014
Created on Wed May 13 23:35:23 2015

@author: federico
"""
import os
import numpy as np
from scipy.signal import argrelextrema
from tkinter import Tk, filedialog
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tifffile as tiff

from tormenta.analysis.stack import Stack


def loadStacks(ask, folder=None):
    # Get filenames from user
    if not(os.path.exists(folder)):
        folder = None
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
    if isinstance(stacksNames, list) or isinstance(stacksNames, tuple):
        pass
    else:
        stacksNames = stacksNames.strip('{}').split('} {')
    folder = os.path.split(stacksNames[0])[0]
    return stacksNames, folder


def beamProfile(ask, folder=None, shape=(512, 512), th=None):

    stacks, folder = loadStacks(ask, folder)

    n = len(stacks)
    profile = np.zeros(shape)
    norm = 0
    for filename in stacks:
        print(filename)
        if filename.endswith('.hdf5'):
            stack = Stack(filename=filename)
            meanFrame = stack.imageData.mean(0)
            stack.close()
        else:
            tfile = tiff.TIFFfile(filename)
            meanFrame = tfile.asarray()
            tfile.close()

        # Beam identification
        hist, edg = np.histogram(meanFrame, bins=100)
        if th is None:
            thres = edg[argrelextrema(hist, np.less)[0][0] + 1]
        else:
            thres = th

        beamMask = np.zeros(shape=meanFrame.shape, dtype=bool)
        beamMask[meanFrame < thres] = True
        beamFrame = np.ma.masked_array(meanFrame, beamMask)

        # Normalization
        meanInt = beamFrame.mean()
        profile += meanFrame / meanInt
        norm += meanInt

    norm /= n

    hist, edg = np.histogram(profile, bins=100)
    if th is None:
        thres = edg[argrelextrema(hist, np.less)[0][0] + 2]
    else:
        thres = th
    beam_mask = np.zeros(shape=profile.shape, dtype=bool)
    beam_mask[profile < thres] = True
    beamProfile = np.ma.masked_array(profile, beam_mask)

    return beamProfile, norm, folder


def frame(image, center=(256, 256), shape=(128, 128)):
    # Untested for not centered frames

    return image[center[0] - int(shape[0] / 2):center[0] + int(shape[0] / 2),
                 center[1] - int(shape[1] / 2):center[1] + int(shape[1] / 2)]


def analyzeBeam(th=None, initialdir=None):
    """
    Script for loading EPI and TIRF images of homogeneous samples for
    measuring the illumination beam profile.
    It loads as many images as you have for both illumination schemes and
    then it calculates the means and correction factors.
    """

    profileEPI, normEPI, folder = beamProfile('Select EPI profiles',
                                              th=th, folder=initialdir)
    profileTIRF, normTIRF, folder = beamProfile('Select TIRF profiles', folder,
                                                th=th)
    TIRFactor = normTIRF / normEPI

    # Measurements in EPI
    EPIFrameFactor = frame(profileEPI).mean() / profileEPI.mean()
    EPIstd = 100 * frame(profileEPI).std() / frame(profileEPI).mean()
    EPIarea = profileTIRF.mask.size - profileEPI.mask.sum()

    # Measurements in TIRF
    TIRFrameFactor = frame(profileTIRF).mean() / profileTIRF.mean()
    TIRstd = 100 * frame(profileTIRF).std() / frame(profileTIRF).mean()
    TIRarea = profileTIRF.mask.size - profileTIRF.mask.sum()

    # Profile images saving
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
    plt.text(800, 200, 'EPI mask area={}'.format(EPIarea) + ' px^2')

    # TIRF profile
    plt.subplot(2, 2, 3)
    plt.imshow(profileTIRF, interpolation='None', cmap=cm.cubehelix)
    plt.title('TIRF profile')
    plt.colorbar()
    plt.text(800, 100,
             'TIRF frame factor={}'.format(np.round(TIRFrameFactor, 2)))
    plt.text(800, 150,
             'TIRF % standard dev={}'.format(np.round(TIRstd, 2)))
    plt.text(800, 200, 'TIRF mask area={}'.format(TIRarea) + ' px^2')
    plt.text(800, 300,
             'TIRF intensity factor={}'.format(np.round(TIRFactor, 2)))

    plt.show()
    area = (TIRarea + EPIarea) / 2
    fFactor = (TIRFrameFactor + EPIFrameFactor) / 2
    return area, fFactor


def intensityCalibration(area, fFactor, objectiveT=0.9, neutralFilter=1000,
                         umPerPx=0.132):

    # Get filenames from user
    try:
        root = Tk()
        dialogTitle = 'Load laser calibration table'
        filename = filedialog.askopenfilename(parent=root, title=dialogTitle)
        root.destroy()
    except OSError:
        print("No files selected!")

    # Data loading
    with open(filename, 'r') as ff:
        titlePlot = ff.readlines()[:2]
        titlePlot = [t.replace('\n', '') for t in titlePlot]
    xlabel = 'photodiode [V]'
    ylabel = 'bfp [mW]'
    tt = np.loadtxt(filename,
                    dtype=[('bfp [mW]', float), ('photodiode [V]', float)],
                    skiprows=3)

    # Fitting and plotting
    coef = np.polyfit(tt['photodiode [V]'], tt['bfp [mW]'], 1)

    # Conversion mW --> kW at the sample
    iFactor = objectiveT * neutralFilter * fFactor / 1000000
    # Conversion px^2 --> cm^2
    area *= (umPerPx * 10**(-4))**2
    # Conversion mW --> kW/cm^2
    factor = iFactor/area
    coef *= factor

    x = np.arange(0, 1.1 * np.max(tt[xlabel]))
    y = np.polynomial.polynomial.polyval(x, coef[::-1])
    plt.scatter(tt[xlabel], tt[ylabel] * factor)
    plt.plot(x, y, 'r', label='V*({0:.2}) + ({1:.2})'.format(coef[0], coef[1]))
    plt.grid()
    plt.title(titlePlot)
    plt.xlabel('Photodiode [V]')
    plt.ylabel('Power [kW/cm^2]')
    plt.legend(loc=2)
    plt.show()

    return coef


def powerCalibration(th=None, initialdir=None):
    area, fFactor = analyzeBeam(th, initialdir)
    return intensityCalibration(area, fFactor)

if __name__ == "__main__":

    coef = powerCalibration(9, initialdir=r'C:\Users\Usuario\Documents\Data')
    print('Coefficients for mW --> kW/cm^2 conversion: ', coef)
