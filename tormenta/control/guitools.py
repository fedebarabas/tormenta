# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:20:02 2015

@author: Federico Barabas
"""
import os
import numpy as np
import configparser
from ast import literal_eval
from tkinter import Tk, filedialog, simpledialog
from lantz import Q_
from PIL import Image
import matplotlib.cm as cm
from scipy.misc import imresize
import tifffile as tiff


# Check for same name conflict
def getUniqueName(name):

    n = 1
    while os.path.exists(name):
        if n > 1:
            name = name.replace('_{}.'.format(n - 1), '_{}.'.format(n))
        else:
            name = insertSuffix(name, '_{}'.format(n))
        n += 1

    return name


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def attrsToTxt(filename, attrs):
    fp = open(filename + '.txt', 'w')
    fp.write('\n'.join('{}= {}'.format(x[0], x[1]) for x in attrs))
    fp.close()


def fileSizeGB(shape):
    # self.nPixels() * self.nExpositions * 16 / (8 * 1024**3)
    return shape[0]*shape[1]*shape[2] / 2**29


def nFramesPerChunk(shape):
    return int(1.8 * 2**29 / (shape[1] * shape[2]))


def getFilename(title, types, initialdir=None):
    try:
        root = Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title=title, filetypes=types,
                                              initialdir=initialdir)
        root.destroy()
        return filename
    except OSError:
        print("No file selected!")


def getFilenames(title, types=[], initialdir=None):
    try:
        root = Tk()
        root.withdraw()
        filenames = filedialog.askopenfilenames(title=title, filetypes=types,
                                                initialdir=initialdir)
        root.destroy()
        return root.tk.splitlist(filenames)
    except OSError:
        print("No files selected!")


# Preset tools
def savePreset(main, filename=None):

    if filename is None:
        root = Tk()
        root.withdraw()
        filename = simpledialog.askstring(title='Save preset',
                                          prompt='Save config file as...')
        root.destroy()

    if filename is None:
        return

    config = configparser.ConfigParser()
    fov = 'Field of view'
    config['Camera'] = {
        'Frame Start': main.frameStart,
        'Shape': main.shape,
        'Shape name': main.tree.p.param(fov).param('Shape').value(),
        'Horizontal readout rate': str(main.HRRatePar.value()),
        'Vertical shift speed': str(main.vertShiftSpeedPar.value()),
        'Clock voltage amplitude': str(main.vertShiftAmpPar.value()),
        'Frame Transfer Mode': str(main.FTMPar.value()),
        'Cropped sensor mode': str(main.cropParam.value()),
        'Set exposure time': str(main.expPar.value()),
        'Pre-amp gain': str(main.PreGainPar.value()),
        'EM gain': str(main.GainPar.value())}

    with open(os.path.join(main.presetDir, filename), 'w') as configfile:
        config.write(configfile)

    main.presetsMenu.addItem(filename)


def loadPreset(main, filename=None):

    tree = main.tree.p
    timings = tree.param('Timings')

    if filename is None:
        preset = main.presetsMenu.currentText()

    config = configparser.ConfigParser()
    config.read(os.path.join(main.presetDir, preset))

    configCam = config['Camera']
    shape = configCam['Shape']

    main.shape = literal_eval(shape)
    main.frameStart = literal_eval(configCam['Frame Start'])

    # Frame size handling
    shapeName = configCam['Shape Name']
    if shapeName == 'Custom':
        main.customFrameLoaded = True
        tree.param('Field of view').param('Shape').setValue(shapeName)
        main.frameStart = literal_eval(configCam['Frame Start'])
        main.adjustFrame()
        main.customFrameLoaded = False
    else:
        tree.param('Field of view').param('Shape').setValue(shapeName)

    vps = timings.param('Vertical pixel shift')
    vps.param('Speed').setValue(Q_(configCam['Vertical shift speed']))

    cva = 'Clock voltage amplitude'
    vps.param(cva).setValue(configCam[cva])

    ftm = 'Frame Transfer Mode'
    timings.param(ftm).setValue(configCam.getboolean(ftm))

    csm = 'Cropped sensor mode'
    if literal_eval(configCam[csm]) is not(None):
        main.cropLoaded = True
        timings.param(csm).param('Enable').setValue(configCam.getboolean(csm))
        main.cropLoaded = False

    hrr = 'Horizontal readout rate'
    timings.param(hrr).setValue(Q_(configCam[hrr]))

    expt = 'Set exposure time'
    timings.param(expt).setValue(float(configCam[expt]))

    pag = 'Pre-amp gain'
    tree.param('Gain').param(pag).setValue(float(configCam[pag]))

    tree.param('Gain').param('EM gain').setValue(int(configCam['EM gain']))


def hideColumn(main):
    if main.hideColumnButton.isChecked():
        main.presetsMenu.hide()
        main.loadPresetButton.hide()
        main.cameraWidget.hide()
        main.viewCtrl.hide()
        main.recWidget.hide()
        main.layout.setColumnMinimumWidth(0, 0)
    else:
        main.presetsMenu.show()
        main.loadPresetButton.show()
        main.cameraWidget.show()
        main.viewCtrl.show()
        main.recWidget.show()
        main.layout.setColumnMinimumWidth(0, 350)


def mouseMoved(main, pos):
    if main.vb.sceneBoundingRect().contains(pos):
        mousePoint = main.vb.mapSceneToView(pos)
        x, y = int(mousePoint.x()), int(main.shape[1] - mousePoint.y())
        main.cursorPos.setText('{}, {}'.format(x, y))
        countsStr = '{} counts'.format(main.image[x, int(mousePoint.y())])
        main.cursorPosInt.setText(countsStr)


def tiff2png(main, filenames=None):

    if filenames is None:
        filenames = getFilenames('Load TIFF files', [('Tiff files', '.tiff'),
                                                     ('Tif files', '.tif')],
                                 main.recWidget.folderEdit.text())

    for filename in filenames:
        with tiff.TiffFile(filename) as tt:
            arr = tt.asarray()
            cmin, cmax = bestLimits(arr)
            arr[arr > cmax] = cmax
            arr[arr < cmin] = cmin
            arr -= arr.min()
            arr = arr/arr.max()

            arr = imresize(arr, (1000, 1000), 'nearest')
            im = Image.fromarray(cm.cubehelix(arr, bytes=True))
            im.save(os.path.splitext(filename)[0] + '.png')


def bestLimits(arr):
    # Best cmin, cmax algorithm taken from ImageJ routine:
    # http://cmci.embl.de/documents/120206pyip_cooking/
    # python_imagej_cookbook#automatic_brightnesscontrast_button
    pixelCount = arr.size
    limit = pixelCount/10
    threshold = pixelCount/5000
    hist, bin_edges = np.histogram(arr, 256)
    i = 0
    found = False
    count = 0
    while True:
        i += 1
        count = hist[i]
        if count > limit:
            count = 0
        found = count > threshold
        if found or i >= 255:
            break
    hmin = i

    i = 256
    while True:
        i -= 1
        count = hist[i]
        if count > limit:
            count = 0
        found = count > threshold
        if found or i < 1:
            break
    hmax = i

    return bin_edges[hmin], bin_edges[hmax]
