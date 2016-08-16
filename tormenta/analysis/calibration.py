# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:19:37 2014
Created on Wed May 13 23:35:23 2015

@author: Federico Barabas
"""
import os
import numpy as np
from tkinter import Tk, filedialog, simpledialog
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tifffile as tiff

from tormenta.analysis.stack import Stack
from tormenta.analysis.gaussians import twoDSymmGaussian


def loadStacks(ask, folder=None):
    # Get filenames from user
    if not(os.path.exists(folder)):
        folder = None
    try:
        root = Tk()
        stacksNames = filedialog.askopenfilenames(parent=root, title=ask,
                                                  initialdir=folder)
        root.destroy()
        # Fix for names with whitespace.
        # Taken from: http://stackoverflow.com/questions/9227859/
        # tkfiledialog-not-converting-results-to-a-python-list-on-windows
        if isinstance(stacksNames, list) or isinstance(stacksNames, tuple):
            pass
        else:
            stacksNames = stacksNames.strip('{}').split('} {')
        folder = os.path.split(stacksNames[0])[0]
        return stacksNames, folder

    except:
        print("No files selected!")


def beamProfile(ask, folder=None, shape=(512, 512), th=None):

    stacks, folder = loadStacks(ask, folder)

    n = len(stacks)
    profile = np.zeros(shape)
    norm = 0
    for filename in stacks:
        print(os.path.split(filename)[1])
        if filename.endswith('.hdf5'):
            stack = Stack(filename=filename)
            meanFrame = stack.imageData.mean(0)
            stack.close()
        else:
            tfile = tiff.TIFFfile(filename)
            meanFrame = tfile.asarray()
            tfile.close()

        # Normalization
        meanInt = meanFrame.mean()
        profile += meanFrame / meanInt
        norm += meanInt

    norm /= n

    # 2D fitting of profile
    try:
        profileFit = twoDSymmGaussian(profile)
        popt, epopt = profileFit.popt, profileFit.epopt
    except RuntimeError as err:
        print("Fitting error: {0}".format(err))
        print("Using sigma = 256.")

        popt, epopt = [0, 0, 0, 256.0], [0, 0, 0, 0]

    # We return the profile, normalization value, sigma and its error
    return profile, norm, (popt[3], epopt[3]), folder


def analyzeBeam(savename, initialdir=None):
    """
    Script for loading EPI and TIRF images of homogeneous samples for
    measuring the illumination beam profile.
    It loads as many images as you have for both illumination schemes and
    then it calculates the means and correction factors.
    """

    beamEPI = beamProfile('Select EPI profiles', folder=initialdir)
    profileEPI, normEPI, sigmaEPI, folder = beamEPI
    beamTIRF = beamProfile('Select TIRF profiles', folder)
    profileTIRF, normTIRF, sigmaTIRF, folder = beamTIRF
    TIRFactor = normTIRF / normEPI

    # STD measurements
    EPIstd = 100 * frame(profileEPI).std() / frame(profileEPI).mean()
    TIRstd = 100 * frame(profileTIRF).std() / frame(profileTIRF).mean()

    # Profile images saving
    im = Image.fromarray(profileEPI)
    im.save(os.path.join(folder, savename + '_profileEPI.tiff'))
    im = Image.fromarray(profileTIRF)
    im.save(os.path.join(folder, savename + '_profileTIRF.tiff'))

    # EPI profile
    plt.subplot(2,  2, 1)
    plt.imshow(profileEPI, interpolation='None', cmap=cm.cubehelix)
    plt.title('EPI profile')
    plt.colorbar()
    plt.text(800, 150, 'EPI % standard dev={}'.format(np.round(EPIstd, 2)))

    # TIRF profile
    plt.subplot(2, 2, 3)
    plt.imshow(profileTIRF, interpolation='None', cmap=cm.cubehelix)
    plt.title('TIRF profile')
    plt.colorbar()
    plt.text(800, 150, 'TIRF % standard dev={}'.format(np.round(TIRstd, 2)))
    tirffactor = np.round(TIRFactor, 2)
    plt.text(800, 300, 'TIRF intensity factor={}'.format(tirffactor))

    plt.savefig(os.path.join(folder, savename + '_profiles.png'),
                bbox_inches='tight')
    plt.show()
    return sigmaEPI, folder


def frame(image, center=(256, 256), shape=(128, 128)):
    # Untested for not centered frames
    return image[center[0] - int(shape[0] / 2):center[0] + int(shape[0] / 2),
                 center[1] - int(shape[1] / 2):center[1] + int(shape[1] / 2)]


def intensityCalibration(sigma, savename, folder=None, objectiveT=0.9,
                         neutralFilter=1, umPerPx=0.12):

    # Get filenames from user
    try:
        root = Tk()
        dialogTitle = 'Load laser calibration table'
        filename = filedialog.askopenfilename(parent=root, title=dialogTitle,
                                              initialdir=folder)
        root.destroy()

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
        iFactor = objectiveT * neutralFilter / 1000000
        # Conversion mW --> kW/cm^2
        factor = iFactor/(2*np.pi*(sigma[0]*umPerPx*0.0001)**2)
        coef *= factor

        x = np.arange(0, 1.1 * np.max(tt[xlabel]))
        y = np.polynomial.polynomial.polyval(x, coef[::-1])
        plt.scatter(tt[xlabel], tt[ylabel] * factor)
        plt.plot(x, y, 'r',
                 label='{0:.3} + V*({1:.3})'.format(coef[1], coef[0]))
        plt.grid()
        plt.title(titlePlot)
        plt.xlabel('Photodiode [V]')
        plt.ylabel('Power [kW/cm^2]')
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        plt.legend(loc=2)
        plt.savefig(os.path.join(folder, savename + '_power'),
                    bbox_inches='tight')
        plt.show()

        return coef

    except:
        print("No files selected!")


def powerCalibration(th=None, initialdir=None):
    root = Tk()
    root.withdraw()
    savename = simpledialog.askstring(title='Saving',
                                      prompt='Save files with prefix...')
    root.destroy()

    sigma, folder = analyzeBeam(savename, initialdir)
    return intensityCalibration(sigma, savename, folder)

if __name__ == "__main__":

    coef = powerCalibration(initialdir=r'C:\Users\Usuario\Documents\Data')
    print('Coefficients for mW --> kW/cm^2 conversion: ', coef)
