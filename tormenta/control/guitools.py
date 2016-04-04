# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:20:02 2015

@author: Federico Barabas
"""
import os
import time
import numpy as np
import h5py as hdf
import tifffile as tiff
import configparser
from ast import literal_eval

from PyQt4 import QtCore
from tkinter import Tk, filedialog, simpledialog
import multiprocessing as mp

from lantz import Q_

import tormenta.analysis.registration as reg


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

    config['Camera'] = {
        'Frame Start': main.frameStart,
        'Shape': main.shape,
        'Shape name': main.tree.p.param('Image frame').param('Shape').value(),
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
        tree.param('Image frame').param('Shape').setValue(shapeName)
        main.frameStart = literal_eval(configCam['Frame Start'])
        main.adjustFrame()
        main.customFrameLoaded = False
    else:
        tree.param('Image frame').param('Shape').setValue(shapeName)

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


# HDF <--> Tiff converter
class TiffConverterThread(QtCore.QThread):

    def __init__(self, filename=None):
        super().__init__()

        self.converter = TiffConverter(filename, self)
        self.converter.moveToThread(self)
        self.started.connect(self.converter.run)
        self.start()


class TiffConverter(QtCore.QObject):

    def __init__(self, filenames, thread, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filenames = filenames
        self.thread = thread

    def run(self):

        self.filenames = getFilenames("Select HDF5 files",
                                      [('HDF5 files', '.hdf5')])

        if len(self.filenames) > 0:
            for filename in self.filenames:
                print('Exporting ', os.path.split(filename)[1])

                file = hdf.File(filename, mode='r')

                for dataname in file:

                    data = file[dataname]
                    filesize = fileSizeGB(data.shape)
                    filename = (os.path.splitext(filename)[0] + '_' + dataname)
                    attrsToTxt(filename, [at for at in data.attrs.items()])

                    if filesize < 2:
                        time.sleep(5)
                        tiff.imsave(filename + '.tiff', data,
                                    description=dataname, software='Tormenta')
                    else:
                        n = nFramesPerChunk(data.shape)
                        i = 0
                        while i < filesize // 1.8:
                            suffix = '_part{}'.format(i)
                            partName = insertSuffix(filename, suffix, '.tiff')
                            tiff.imsave(partName, data[i*n:(i + 1)*n],
                                        description=dataname,
                                        software='Tormenta')
                            i += 1
                        if filesize % 2 > 0:
                            suffix = '_part{}'.format(i)
                            partName = insertSuffix(filename, suffix, '.tiff')
                            tiff.imsave(partName, data[i*n:],
                                        description=dataname,
                                        software='Tormenta')

                file.close()
                print('done')

        self.filenames = None
        self.thread.terminate()
        # for opening attributes this should work:
        # myprops = dict(line.strip().split('=') for line in
        #                open('/Path/filename.txt'))


class HtransformStack(QtCore.QObject):
    """ Transforms all frames of channel 1 using matrix H."""

    finished = QtCore.pyqtSignal()

    def run(self):
        Hname = getFilename("Select affine transformation matrix",
                            [('npy files', '.npy')])
        H = np.load(Hname)
        filenames = getFilenames("Select files for affine transformation",
                                 initialdir=os.path.split(Hname)[0])
        for filename in filenames:

            ext = os.path.splitext(filename)[1]

            if ext == '.hdf5':
                filename2 = insertSuffix(filename, '_corrected')
                with hdf.File(filename, 'r') as f0, \
                        hdf.File(filename2, 'w') as f1:

                    print(time.strftime("%Y-%m-%d %H:%M:%S") +
                          ' Transforming stack ' + os.path.split(filename)[1])

                    dat0 = f0['data']
                    n = len(dat0)
                    f1.create_dataset(name='data',
                                      data=np.append(dat0[:, :128, :],
                                                     np.zeros((n, 128, 266),
                                                              dtype=np.uint16),
                                                     1))

                    # Multiprocessing
                    cpus = mp.cpu_count()
                    step = n // cpus
                    chunks = [[i*step, (i + 1)*step] for i in np.arange(cpus)]
                    chunks[-1][1] = n
                    args = [[dat0[i:j, -128:, :], H] for i, j in chunks]
                    pool = mp.Pool(processes=cpus)
                    results = pool.map(transformChunk, args)
                    pool.close()
                    pool.join()
                    f1['data'][:, -128:, :] = np.concatenate(results[:])

                    print(time.strftime("%Y-%m-%d %H:%M:%S") + ' done')

            elif ext in ['.tiff', '.tif']:
                with tiff.TiffFile(filename) as tt:
                    data = tt.asarray()
                    tiff.imsave(insertSuffix(filename, '_ch0'), data[:128, :])
                    tiff.imsave(insertSuffix(filename, '_ch1'),
                                reg.h_affine_transform(data[-128:, :], H))

        self.finished.emit()


def transformChunk(args):

    data, H = args

    n = len(data)
    out = np.zeros((n, 128, 266), dtype=np.uint16)
    for f in np.arange(n):
        out[f] = reg.h_affine_transform(data[f], H)

    return out
