# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:20:02 2015

@author: federico
"""
import os
import time
import pyqtgraph as pg
import numpy as np
import h5py as hdf
import tifffile as tiff
import configparser
from ast import literal_eval

from PyQt4 import QtCore, QtGui
from tkinter import Tk, filedialog, simpledialog

from lantz import Q_


# taken from https://www.mrao.cam.ac.uk/~dag/CUBEHELIX/cubehelix.py
def cubehelix(gamma=1.0, s=0.5, r=-1.5, h=1.0):
    def get_color_function(p0, p1):
        def color(x):
            xg = x ** gamma
            a = h * xg * (1 - xg) / 2
            phi = 2 * np.pi * (s / 3 + r * x)
            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    array = np.empty((256, 3))
    abytes = np.arange(0, 1, 1/256.)
    array[:, 0] = get_color_function(-0.14861, 1.78277)(abytes) * 255
    array[:, 1] = get_color_function(-0.29227, -0.90649)(abytes) * 255
    array[:, 2] = get_color_function(1.97294, 0.0)(abytes) * 255
    return array


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


def attrsToTxt(filename, attrs):
    fp = open(filename + '.txt', 'w')
    fp.write('\n'.join('{}= {}'.format(x[0], x[1]) for x in attrs))
    fp.close()


def insertSuffix(filename, suffix, newExt=None):
    names = os.path.splitext(filename)
    if newExt is None:
        return names[0] + suffix + names[1]
    else:
        return names[0] + suffix + newExt


def fileSizeGB(shape):
    # self.nPixels() * self.nExpositions * 16 / (8 * 1024**3)
    return shape[0]*shape[1]*shape[2] / 2**29


def nFramesPerChunk(shape):
    return int(1.8 * 2**29 / (shape[1] * shape[2]))


def getFilenames(title, filetypes):
    try:
        root = Tk()
        root.withdraw()
        filenames = filedialog.askopenfilenames(title=title,
                                                filetypes=filetypes)
        root.destroy()
        return root.tk.splitlist(filenames)
    except OSError:
        print("No files selected!")


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

        if self.filenames is None:
            self.filenames = getFilenames("Select HDF5 files",
                                          [('HDF5 files', '.hdf5')])

        else:
            self.filenames = [self.filenames]

        if len(self.filenames) > 0:
            for filename in self.filenames:

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

        print(self.filenames, 'exported to TIFF')
        self.filenames = None
        self.thread.terminate()
        # for opening attributes this should work:
        # myprops = dict(line.strip().split('=') for line in
        #                open('/Path/filename.txt'))


class Grid():

    def __init__(self, viewBox, shape):

        self.showed = False
        self.vb = viewBox
        self.shape = shape
        pen = QtGui.QPen(QtCore.Qt.yellow, 0.75, QtCore.Qt.DotLine)
        pen2 = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)

        self.yline1 = pg.InfiniteLine(pos=0.25*self.shape[0], pen=pen)
        self.yline2 = pg.InfiniteLine(pos=0.375*self.shape[0], pen=pen)
        self.yline3 = pg.InfiniteLine(pos=0.50*self.shape[0], pen=pen2)
        self.yline4 = pg.InfiniteLine(pos=0.625*self.shape[0], pen=pen)
        self.yline5 = pg.InfiniteLine(pos=0.75*self.shape[0], pen=pen)
        self.xline1 = pg.InfiniteLine(pos=0.25*self.shape[1], pen=pen, angle=0)
        self.xline2 = pg.InfiniteLine(pos=0.375*self.shape[1], pen=pen2,
                                      angle=0)
        self.xline3 = pg.InfiniteLine(pos=0.50*self.shape[1], pen=pen2,
                                      angle=0)
        self.xline4 = pg.InfiniteLine(pos=0.625*self.shape[1], pen=pen2,
                                      angle=0)
        self.xline5 = pg.InfiniteLine(pos=0.75*self.shape[1], pen=pen, angle=0)

    def update(self, shape):
        self.yline1.setPos(0.25*shape[0])
        self.yline2.setPos(0.375*shape[0])
        self.yline3.setPos(0.50*shape[0])
        self.yline4.setPos(0.625*shape[0])
        self.yline5.setPos(0.75*shape[0])
        self.xline1.setPos(0.25*shape[1])
        self.xline2.setPos(0.375*shape[1])
        self.xline3.setPos(0.50*shape[1])
        self.xline4.setPos(0.625*shape[1])
        self.xline5.setPos(0.75*shape[1])

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.addItem(self.xline1)
        self.vb.addItem(self.xline2)
        self.vb.addItem(self.xline3)
        self.vb.addItem(self.xline4)
        self.vb.addItem(self.xline5)
        self.vb.addItem(self.yline1)
        self.vb.addItem(self.yline2)
        self.vb.addItem(self.yline3)
        self.vb.addItem(self.yline4)
        self.vb.addItem(self.yline5)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.xline1)
        self.vb.removeItem(self.xline2)
        self.vb.removeItem(self.xline3)
        self.vb.removeItem(self.xline4)
        self.vb.removeItem(self.xline5)
        self.vb.removeItem(self.yline1)
        self.vb.removeItem(self.yline2)
        self.vb.removeItem(self.yline3)
        self.vb.removeItem(self.yline4)
        self.vb.removeItem(self.yline5)
        self.showed = False


class TwoColorGrid():

    def __init__(self, viewBox, shape=(512, 512)):

        self.showed = False
        self.vb = viewBox
        self.shape = shape

        pen = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)
        pen2 = QtGui.QPen(QtCore.Qt.yellow, 0.75, QtCore.Qt.DotLine)

        self.rectT = QtGui.QGraphicsRectItem(192, 118, 128, 128)
        self.rectT.setPen(pen)
        self.rectR = QtGui.QGraphicsRectItem(192, 266, 128, 128)
        self.rectR.setPen(pen)
        self.yLine = pg.InfiniteLine(pos=0.5*self.shape[0], pen=pen2)
        self.xLine = pg.InfiniteLine(pos=0.5*self.shape[1], pen=pen2, angle=0)
        self.xLineT = pg.InfiniteLine(pos=182, pen=pen2, angle=0)
        self.xLineR = pg.InfiniteLine(pos=330, pen=pen2, angle=0)

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.addItem(self.rectT)
        self.vb.addItem(self.rectR)
        self.vb.addItem(self.yLine)
        self.vb.addItem(self.xLine)
        self.vb.addItem(self.xLineR)
        self.vb.addItem(self.xLineT)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.rectT)
        self.vb.removeItem(self.rectR)
        self.vb.removeItem(self.yLine)
        self.vb.removeItem(self.xLine)
        self.vb.removeItem(self.xLineR)
        self.vb.removeItem(self.xLineT)
        self.showed = False


class Crosshair():

    def __init__(self, viewBox):

        self.showed = False

        self.vLine = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.hLine = pg.InfiniteLine(pos=0, angle=0,  movable=False)
        self.vb = viewBox

    def mouseMoved(self, pos):
        if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def mouseClicked(self):
        try:
            self.vb.scene().sigMouseMoved.disconnect(self.mouseMoved)
        except:
            pass

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.scene().sigMouseClicked.connect(self.mouseClicked)
        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
        self.vb.addItem(self.vLine, ignoreBounds=False)
        self.vb.addItem(self.hLine, ignoreBounds=False)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.vLine)
        self.vb.removeItem(self.hLine)
        self.showed = False


class ROI(pg.ROI):

    def __init__(self, shape, vb, pos, handlePos, handleCenter, *args,
                 **kwargs):

        self.mainShape = shape

        pg.ROI.__init__(self, pos, size=(128, 128), pen='y', *args, **kwargs)
        self.addScaleHandle(handlePos, handleCenter, lockAspect=True)
        vb.addItem(self)

        self.label = pg.TextItem()
        self.label.setPos(self.pos()[0] + self.size()[0],
                          self.pos()[1] + self.size()[1])
        self.label.setText('128x128')

        self.sigRegionChanged.connect(self.updateText)

        vb.addItem(self.label)

    def updateText(self):
        self.label.setPos(self.pos()[0] + self.size()[0],
                          self.pos()[1] + self.size()[1])
        size = np.round(self.size()).astype(np.int)
        self.label.setText('{}x{}'.format(size[0], size[1]))

    def hide(self, *args, **kwargs):
        super().hide(*args, **kwargs)
        self.label.hide()


class cropROI(pg.ROI):

    def __init__(self, shape, vb, *args, **kwargs):

        self.mainShape = shape

        pg.ROI.__init__(self, pos=(shape[0], shape[1]), size=(128, 128),
                        scaleSnap=True, translateSnap=True, movable=False,
                        pen='y', *args, **kwargs)
        self.addScaleHandle((0, 1), (1, 0))
