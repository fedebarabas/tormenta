# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@authors: Federico Barabas, Luciano Masullo
"""

import subprocess
import sys
import numpy as np
import os
import datetime
import time
import re
from tkinter import Tk, filedialog, messagebox
import h5py as hdf
import tifffile as tiff     # http://www.lfd.uci.edu/~gohlke/pythonlibs/#vlfd
from lantz import Q_
import lantz.log

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget

# tormenta imports
from tormenta.control.filter_table import FilterTable
import tormenta.utils as utils
import tormenta.control.lasercontrol as lasercontrol
import tormenta.control.focus as focus
import tormenta.control.molecules_counter as moleculesCounter
import tormenta.control.ontime as ontime
import tormenta.control.guitools as guitools
import tormenta.control.pyqtsubclasses as pyqtsub
import tormenta.control.viewbox_tools as viewbox_tools
import tormenta.analysis.registration as reg
import tormenta.analysis.stack as stack


class RecordingWidget(QtGui.QFrame):

    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.dataname = 'data'      # In case I need a QLineEdit for this
        self.initialDir = r'C:\Users\Usuario\Documents\Data'

        self.Hname = None
        self.H = None
        self.corrShape = None
        self.cropShape = None
        self.xlim = None
        self.ylim = None
        self.shape = self.main.shape

        # Title
        recTitle = QtGui.QLabel('<h2><strong>Recording</strong></h2>')
        recTitle.setTextFormat(QtCore.Qt.RichText)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Folder and filename fields
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        openFolderButton = QtGui.QPushButton('Open')
        openFolderButton.clicked.connect(self.openFolder)
        loadFolderButton = QtGui.QPushButton('Browse')
        loadFolderButton.clicked.connect(self.loadFolder)
        self.filenameEdit = QtGui.QLineEdit('filename')

        # Snap button
        self.snapButton = QtGui.QPushButton('Snap')
        self.snapButton.setStyleSheet("font-size:16px")
        self.snapButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.snapButton.setToolTip('Ctrl+S')
        self.snapButton.clicked.connect(self.snap)

        # REC button
        self.recButton = QtGui.QPushButton('REC')
        self.recButton.setStyleSheet("font-size:16px")
        self.recButton.setCheckable(True)
        self.recButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                     QtGui.QSizePolicy.Expanding)
        self.recButton.setToolTip('Ctrl+R')
        self.recButton.clicked.connect(self.waitForSignal)
        self.recShortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+R'), self)
        self.recShortcut.setEnabled(False)

        # Recording format
        self.recFormat = QtGui.QComboBox(self)
        self.recFormat.addItem('tiff')
        self.recFormat.addItem('hdf5')
        self.recFormat.setFixedWidth(50)

        # Number of frames and measurement timing
        self.currentFrame = QtGui.QLabel('0 /')
        self.currentFrame.setAlignment((QtCore.Qt.AlignRight |
                                        QtCore.Qt.AlignVCenter))
        self.currentFrame.setFixedWidth(45)
        self.numExpositionsEdit = QtGui.QLineEdit('100')
        self.numExpositionsEdit.setFixedWidth(45)
        self.tRemaining = QtGui.QLabel()
        self.tRemaining.setStyleSheet("font-size:14px")
        self.tRemaining.setAlignment((QtCore.Qt.AlignCenter |
                                      QtCore.Qt.AlignVCenter))
        self.numExpositionsEdit.textChanged.connect(self.nChanged)
        self.updateRemaining()

        self.progressBar = QtGui.QProgressBar()
        self.progressBar.setTextVisible(False)
        self.progressBar.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                       QtGui.QSizePolicy.Expanding)
        self.fileSizeLabel = QtGui.QLabel()
        self.nChanged()

        # Recording buttons layout
        buttonWidget = QtGui.QWidget()
        buttonGrid = QtGui.QGridLayout()
        buttonWidget.setLayout(buttonGrid)
        buttonGrid.addWidget(self.snapButton, 0, 0)
        buttonWidget.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                   QtGui.QSizePolicy.Expanding)
        buttonGrid.addWidget(self.recButton, 0, 2)
        buttonGrid.addWidget(self.recFormat, 0, 3)

        recGrid = QtGui.QGridLayout()
        self.setLayout(recGrid)

        recGrid.addWidget(recTitle, 0, 0, 1, 3)
        recGrid.addWidget(QtGui.QLabel('Folder'), 1, 0)
        recGrid.addWidget(loadFolderButton, 1, 4)
        recGrid.addWidget(openFolderButton, 1, 3)
        recGrid.addWidget(self.folderEdit, 2, 0, 1, 5)
        recGrid.addWidget(QtGui.QLabel('Filename'), 3, 0, 1, 2)
        recGrid.addWidget(self.filenameEdit, 4, 0, 1, 5)
        recGrid.addWidget(QtGui.QLabel('Number of expositions'), 5, 0)
        recGrid.addWidget(self.currentFrame, 5, 1)
        recGrid.addWidget(self.numExpositionsEdit, 5, 2)
        recGrid.addWidget(self.progressBar, 5, 3, 2, 2)
        recGrid.addWidget(self.tRemaining, 5, 3, 2, 2)
        recGrid.addWidget(QtGui.QLabel('File size'), 6, 0)
        recGrid.addWidget(self.fileSizeLabel, 6, 2)
        recGrid.addWidget(buttonWidget, 7, 0, 1, 5)

        recGrid.setColumnMinimumWidth(0, 70)
        recGrid.setRowMinimumHeight(7, 40)

        self.writable = True
        self.readyToRecord = False

    @property
    def readyToRecord(self):
        return self._readyToRecord

    @readyToRecord.setter
    def readyToRecord(self, value):
        self.snapButton.setEnabled(value)
        self.recButton.setEnabled(value)
        self.recShortcut.setEnabled(value)
        self._readyToRecord = value

    @property
    def writable(self):
        return self._writable

    @writable.setter
    def writable(self, value):
        self.folderEdit.setEnabled(value)
        self.filenameEdit.setEnabled(value)
        self.numExpositionsEdit.setEnabled(value)
        self.recFormat.setEditable(value)
        self._writable = value

    def n(self):
        text = self.numExpositionsEdit.text()
        if text == '':
            return 0
        else:
            return int(text)

    def nChanged(self):
        self.updateRemaining()
#        self.limitExpositions(9)
        shape = [self.n(), self.shape[0], self.shape[1]]
        size = guitools.fileSizeGB(shape)
        self.fileSizeLabel.setText("{0:.2f} GB".format(size))

    def updateRemaining(self):
        rSecs = self.main.t_acc_real.magnitude * self.n()
        rTime = datetime.timedelta(seconds=np.round(rSecs))
        self.tRemaining.setText('{}'.format(rTime))

    def nPixels(self):
        return self.main.shape[0] * self.main.shape[1]

    # Setting a xGB limit on file sizes to be able to open them in Fiji
    def limitExpositions(self, xGB):
        # nMax = xGB * 8 * 1024**3 / (pixels * 16)
        nMax = xGB * 2**29 / self.nPixels()
        if self.n() > nMax:
            self.numExpositionsEdit.setText(str(np.round(nMax).astype(int)))

    def openFolder(self, path):
        if sys.platform == 'darwin':
            subprocess.check_call(['open', '', self.folderEdit.text()])
        elif sys.platform == 'linux':
            subprocess.check_call(['gnome-open', '', self.folderEdit.text()])
        elif sys.platform == 'win32':
            os.startfile(self.folderEdit.text())

    def loadFolder(self):
        try:
            root = Tk()
            root.withdraw()
            folder = filedialog.askdirectory(parent=root,
                                             initialdir=self.initialDir)
            root.destroy()
            if folder != '':
                self.folderEdit.setText(folder)
        except OSError:
            pass

    # Attributes saving
    def getAttrs(self):
        attrs = self.main.tree.attrs()
        um = self.main.umxpx
        attrs.extend([('Date', time.strftime("%Y-%m-%d")),
                      ('Start time', time.strftime("%H:%M:%S")),
                      ('element_size_um', (1, um, um)),
                      ('NA', 1.42),
                      ('lambda_em', 670),
                      ('Affine matrix filename', self.Hname)])
        for c in self.main.laserWidgets.controls:
            name = re.sub('<[^<]+?>', '', c.name.text())
            attrs.extend([(name+'_power', c.laser.power),
                          (name+'_intensity', c.intensityEdit.text()),
                          (name+'_calibrated', c.calibratedCheck.isChecked())])
        return attrs

    def loadH(self):
        self.Hname = utils.getFilename('Load affine matrix',
                                       [('Numpy arrays', '.npy')],
                                       self.folderEdit.text())
        self.H = np.load(self.Hname)
        self.xlim, self.ylim, self.cropShape = reg.get_affine_shapes(self.H)

    def snap(self):

        folder = self.folderEdit.text()
        if os.path.exists(folder):

            image = self.main.andor.most_recent_image16(self.main.shape)
            time.sleep(0.01)

            dim = (self.main.umxpx * np.array(self.main.shape)).astype(np.int)
            sh = str(dim[0]) + 'x' + str(dim[1])
            rootname = os.path.join(folder, self.filenameEdit.text()) + '_wf'
            savename = rootname + sh + '.tif'
            savename = guitools.getUniqueName(savename)
            image = np.flipud(image.astype(np.uint16))
            tiff.imsave(savename, image, software='Tormenta', imagej=True,
                        resolution=(1/self.main.umxpx, 1/self.main.umxpx),
                        metadata={'spacing': 1, 'unit': 'um'})
            guitools.attrsToTxt(os.path.splitext(savename)[0], self.getAttrs())

            # Two-color-corrected snap saving
            imageFramePar = self.main.tree.p.param('Field of view')
            shapeStr = imageFramePar.param('Shape').value()
            twoColors = shapeStr.startswith('Two-colors')
            if twoColors and (self.H is not None):
                side = int(shapeStr.split()[1][:-2])

                # Corrected image
                im0 = image[:side, :]
                im1 = reg.h_affine_transform(image[-side:, :], self.H)

                dim = (self.main.umxpx * np.array(im0.shape)).astype(np.int)
                sh = str(dim[0]) + 'x' + str(dim[1])
                corrName = rootname + sh + '.tif'
                tiff.imsave(utils.insertSuffix(corrName, '_corrected_ch0'),
                            im0, software='Tormenta', imagej=True,
                            resolution=(1/self.main.umxpx, 1/self.main.umxpx),
                            metadata={'spacing': 1, 'unit': 'um'})

                tiff.imsave(utils.insertSuffix(corrName, '_corrected_ch1'),
                            im1, software='Tormenta', imagej=True,
                            resolution=(1/self.main.umxpx, 1/self.main.umxpx),
                            metadata={'spacing': 1, 'unit': 'um'})

                # Corrected and cropped image
                crop2chShape = (2*self.cropShape[0], self.cropShape[1])
                im0c = im0[self.xlim[0]:self.xlim[1],
                           self.ylim[0]:self.ylim[1]]
                im1c = im1[self.xlim[0]:self.xlim[1],
                           self.ylim[0]:self.ylim[1]]

                dim = (self.main.umxpx * np.array(crop2chShape)).astype(np.int)
                sh = str(dim[0]) + 'x' + str(dim[1])
                cropName = rootname + sh + '.tif'
                ch0Name = utils.insertSuffix(cropName, '_corrected_crop_ch0')
                tiff.imsave(ch0Name, im0c, software='Tormenta', imagej=True,
                            resolution=(1/self.main.umxpx, 1/self.main.umxpx),
                            metadata={'spacing': 1, 'unit': 'um'})
                ch1Name = utils.insertSuffix(cropName, '_corrected_crop_ch1')
                tiff.imsave(ch1Name, im1c, software='Tormenta', imagej=True,
                            resolution=(1/self.main.umxpx, 1/self.main.umxpx),
                            metadata={'spacing': 1, 'unit': 'um'})
        else:
            self.folderWarning()

    def folderWarning(self):
        root = Tk()
        root.withdraw()
        messagebox.showwarning(title='Warning', message="Folder doesn't exist")
        root.destroy()

    def tiffWarning(self):
        root = Tk()
        root.withdraw()
        message = "Tiff format size limit is 2GB"
        messagebox.showwarning(title='Warning', message=message)
        root.destroy()

    def updateGUI(self, image):
        self.main.img.setImage(image, autoLevels=False)

        if self.main.moleculeWidget.enabled:
            self.main.moleculeWidget.graph.update(image)

        if self.main.crosshair.showed:
                ycoord = int(np.round(self.crosshair.hLine.pos()[1]))
                xcoord = int(np.round(self.crosshair.vLine.pos()[0]))
                self.xProfile.setData(image[:, ycoord])
                self.yProfile.setData(image[xcoord])

        # fps calculation
        self.main.fpsMath()

        # Elapsed and remaining times and frames
        eSecs = np.round(ptime.time() - self.startTime)
        nframe = self.worker.j
        rFrames = self.n() - nframe
        rSecs = np.round(self.main.t_acc_real.magnitude * rFrames)
        rText = '{}'.format(datetime.timedelta(seconds=rSecs))
        self.tRemaining.setText(rText)
        self.currentFrame.setText(str(nframe) + ' /')
        self.progressBar.setValue(100*(1 - rSecs / (eSecs + rSecs)))

    # This is the function triggered by the self.main.recShortcut shortcut
    def startRecKey(self):

        if self.recButton.isChecked():
            self.recButton.setChecked(False)

        else:
            self.recButton.setChecked(True)
            self.waitForSignal()

    # Waits for the signal indicating intensity measurement has finished
    def waitForSignal(self):
        laserWidgets = self.main.laserWidgets
        laserWidgets.worker.doneSignal.connect(self.startRecording)
        powerChanged = [c.powerChanged for c in laserWidgets.controls]
        if np.any(np.array(powerChanged)):
            laserWidgets.getIntensities()
        else:
            laserWidgets.worker.doneSignal.emit()

    def startRecording(self):

        if self.recButton.isChecked():

            folder = self.folderEdit.text()
            recFormat = self.recFormat.currentText()
            shape = (self.n(), self.main.shape[0], self.main.shape[1])

            if not(os.path.exists(folder)):
                self.folderWarning()
                self.recButton.setChecked(False)

            elif guitools.fileSizeGB(shape) > 2 and recFormat == 'tiff':
                self.tiffWarning()
                self.recButton.setChecked(False)

            else:
                self.writable = False
                self.readyToRecord = False
                self.recButton.setEnabled(True)
                self.recButton.setText('STOP')
                self.main.tree.writable = False
                self.main.liveviewButton.setEnabled(False)
                self.main.viewtimer.stop()

                name = os.path.join(folder, self.filenameEdit.text())
                self.startTime = ptime.time()

                imageFramePar = self.main.tree.p.param('Field of view')
                shapeStr = imageFramePar.param('Shape').value()
                twoColors = shapeStr.startswith('Two-colors')
                twoColors = twoColors and (self.H is not None)
                self.worker = RecWorker(self.main.andor, self.main.umxpx,
                                        shape, self.main.t_exp_real, name,
                                        recFormat, self.dataname,
                                        self.getAttrs(), twoColors, self.H,
                                        self.cropShape, self.xlim, self.ylim)
                self.worker.updateSignal.connect(self.updateGUI)
                self.recordingThread = QtCore.QThread(self)
                self.worker.moveToThread(self.recordingThread)
                self.worker.doneSignal.connect(self.endRecording)
                self.worker.doneSignal.connect(self.recordingThread.quit)
                self.worker.doneSignal.connect(self.worker.deleteLater)
                self.recordingThread.started.connect(self.worker.start)
                self.recordingThread.start()

        else:
            self.worker.pressed = False

    def endRecording(self):

        # Attenuate excitation x1000, turn blue laser off
        if self.main.flipAfter.isChecked():
            self.main.flipperInPath(True)
        if self.main.uvOff.isChecked():
            self.main.laserWidgets.blueControl.laser.power_sp = Q_(0, 'mW')

        if self.main.focusWidget.focusDataBox.isChecked():
            self.main.focusWidget.exportData()
        else:
            self.main.focusWidget.graph.savedDataSignal = []
            self.main.focusWidget.graph.mean = 0
            self.main.focusWidget.n = 1
            self.main.focusWidget.max_dev = 0

        def converterFunction():
            return pyqtsub.TiffConverterThread(self.savename)

        self.main.exportlastAction.triggered.connect(converterFunction)
        self.main.exportlastAction.setEnabled(True)

        self.writable = True
        self.readyToRecord = True
        self.recButton.setText('REC')
        self.recButton.setChecked(False)
        self.main.tree.writable = True
        self.main.liveviewButton.setEnabled(True)
        self.main.liveviewStart(update=False)
        self.main.laserWidgets.worker.doneSignal.disconnect()
        for c in self.main.laserWidgets.controls:
            c.powerChanged = False


class RecWorker(QtCore.QObject):

    updateSignal = QtCore.pyqtSignal(np.ndarray)
    doneSignal = QtCore.pyqtSignal()

    def __init__(self, andor, umPerPx, shape, t_exp, savename, fileformat,
                 dataname, attrs, twoColors, H, cropShape, xlim, ylim,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.andor = andor
        self.umxpx = umPerPx
        self.shape = shape
        self.t_exp = t_exp
        self.savename = savename
        self.recFormat = fileformat
        self.dataname = dataname
        self.attrs = attrs
        self.pressed = True
        self.frameShape = (self.shape[1], self.shape[2])
        self.n = self.shape[0]

        self.twoColors = twoColors
        self.H = H
        self.cropShape = cropShape
        try:
            self.corrShape = (self.n, 2*self.cropShape[0], self.cropShape[1])
        except:
            self.corrShape = None
        self.xlim = xlim
        self.ylim = ylim

    def start(self):

        # Acquisition preparation
        if self.andor.status != 'Camera is idle, waiting for instructions.':
            self.andor.abort_acquisition()
        else:
            self.andor.shutter(0, 1, 0, 0, 0)

        # Frame counter
        self.j = 0

        self.andor.free_int_mem()
        self.andor.acquisition_mode = 'Kinetics'
        self.andor.set_n_kinetics(self.shape[0])
        self.andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp.magnitude, 1)))

        self.savename = (self.savename + '.' + self.recFormat)
        self.savename = guitools.getUniqueName(self.savename)

        if self.recFormat == 'tiff':

            self.tags = [('resolution_unit', 'H', 1, 3, True)]
            self.resolution = (1/self.umxpx, 1/self.umxpx)

            if not(self.twoColors):
                self.singleColorTIFF()
            else:
                self.twoColorTIFF()

        elif self.recFormat == 'hdf5':
            if not(self.twoColors):
                self.singleColorHDF5()
            else:
                self.twoColorHDF5()

        self.doneSignal.emit()

    def singleColorTIFF(self):

        with tiff.TiffWriter(self.savename, software='Tormenta') as storeFile:

            while self.j < self.shape[0] and self.pressed:
                time.sleep(self.t_exp.magnitude)
                if self.andor.n_images_acquired > self.j:
                    i, self.j = self.andor.new_images_index
                    newImages = self.andor.images16(i, self.j, self.frameShape,
                                                    1, self.n)
                    self.updateSignal.emit(np.transpose(newImages[-1]))

                    newData = newImages[:, ::-1].astype(np.uint16)

                    # This is done frame by frame in order to have a
                    # contiguously saved tiff so it's easily opened in ImageJ
                    # or in python through tifffile
                    for frame in newData:
                        storeFile.save(frame, photometric='minisblack',
                                       resolution=self.resolution,
                                       extratags=self.tags)

        # Saving parameters
        metaName = os.path.splitext(self.savename)[0] + '_metadata.hdf5'
        with hdf.File(metaName, "w") as metaFile:
            for item in self.attrs:
                if item[1] is not None:
                    metaFile[item[0]] = item[1]

    def twoColorTIFF(self):

        cropName = utils.insertSuffix(self.savename, '_corrected')

        with tiff.TiffWriter(self.savename, software='Tormenta') as storeFile,\
                tiff.TiffWriter(cropName, software='Tormenta') as cropFile:

            while self.j < self.shape[0] and self.pressed:

                time.sleep(self.t_exp.magnitude)
                if self.andor.n_images_acquired > self.j:
                    i, self.j = self.andor.new_images_index
                    newImages = self.andor.images16(i, self.j, self.frameShape,
                                                    1, self.n)
                    self.updateSignal.emit(np.transpose(newImages[-1]))
                    newData = newImages[:, ::-1]

                    # This is done frame by frame in order to have contiguously
                    # saved tiff files so they're easily opened in ImageJ
                    # or in python through tifffile
                    for frame in newData:
                        # Corrected image
                        im0 = frame[:128, :]
                        im1 = reg.h_affine_transform(frame[-128:, :], self.H)

                        # Corrected and cropped image
                        im1c = im1[self.xlim[0]:self.xlim[1],
                                   self.ylim[0]:self.ylim[1]]
                        im0c = im0[self.xlim[0]:self.xlim[1],
                                   self.ylim[0]:self.ylim[1]]
                        imc = np.hstack((im0c, im1c)).astype(np.uint16)
                        cropFile.save(imc, photometric='minisblack',
                                      resolution=self.resolution,
                                      extratags=self.tags)

                        # Raw image
                        storeFile.save(frame, photometric='minisblack',
                                       resolution=self.resolution,
                                       extratags=self.tags)

        # Saving parameters
        metaName = os.path.splitext(self.savename)[0] + '_metadata.hdf5'
        corrMetaName = utils.insertSuffix(metaName, '_corrected')
        with hdf.File(metaName, "w") as metaFile, \
                hdf.File(corrMetaName, "w") as corrMetaFile:
            for item in self.attrs:
                if item[1] is not None:
                    metaFile[item[0]] = item[1]
                    corrMetaFile[item[0]] = item[1]
            corrMetaFile.create_dataset(name='Affine matrix', data=self.H)

    def singleColorHDF5(self):

        with hdf.File(self.savename, "w") as storeFile:
            storeFile.create_dataset(name=self.dataname, shape=self.shape,
                                     maxshape=self.shape, dtype=np.uint16)
            dataset = storeFile[self.dataname]

            while self.j < self.shape[0] and self.pressed:

                time.sleep(self.t_exp.magnitude)
                if self.andor.n_images_acquired > self.j:
                    i, self.j = self.andor.new_images_index
                    newImages = self.andor.images16(i, self.j, self.frameShape,
                                                    1, self.n)
                    self.updateSignal.emit(np.transpose(newImages[-1]))
                    dataset[i - 1:self.j] = newImages[:, ::-1]

            # Crop dataset if it's stopped before finishing
            if self.j < self.shape[0]:
                dataset.resize((self.j, self.shape[1], self.shape[2]))

            # Saving parameters
            for item in self.attrs:
                if item[1] is not None:
                    dataset.attrs[item[0]] = item[1]

    def twoColorHDF5(self):

        cropSavename = utils.insertSuffix(self.savename, '_corrected')

        with hdf.File(self.savename, "w") as storeFile, \
                hdf.File(cropSavename, "w") as cropStoreFile:

            storeFile.create_dataset(name=self.dataname, shape=self.shape,
                                     maxshape=self.shape, dtype=np.uint16)
            dataset = storeFile[self.dataname]

            cropStoreFile.create_dataset(name=self.dataname,
                                         shape=self.corrShape,
                                         maxshape=self.corrShape,
                                         dtype=np.uint16)
            cropDataset = cropStoreFile[self.dataname]

            while self.j < self.shape[0] and self.pressed:

                time.sleep(self.t_exp.magnitude)
                if self.andor.n_images_acquired > self.j:
                    i, self.j = self.andor.new_images_index
                    newImages = self.andor.images16(i, self.j, self.frameShape,
                                                    1, self.n)
                    self.updateSignal.emit(np.transpose(newImages[-1]))
                    data = newImages[:, ::-1]

                    # Corrected image
                    im0 = data[:, :128, :]
                    im1 = np.zeros(data[:, -128:, :].shape)
                    for k in np.arange(len(im1)):
                        im1[k] = reg.h_affine_transform(data[k, -128:, :],
                                                        self.H)

                    # Corrected and cropped image
                    im0c = im0[:, self.xlim[0]:self.xlim[1],
                               self.ylim[0]:self.ylim[1]]
                    im1c = im1[:, self.xlim[0]:self.xlim[1],
                               self.ylim[0]:self.ylim[1]]
                    cropDataset[i - 1:self.j, :self.cropShape[0], :] = im0c
                    cropDataset[i - 1:self.j, self.cropShape[0]:, :] = im1c

            # Crop dataset if it's stopped before finishing
            if self.j < self.shape[0]:
                dataset.resize((self.j, self.shape[1], self.shape[2]))
                newCropShape = (self.j, self.corrShape[1], self.corrShape[2])
                cropDataset.resize(newCropShape)

            # Saving parameters
            for item in self.attrs:
                if item[1] is not None:
                    dataset.attrs[item[0]] = item[1]
                    cropDataset.attrs[item[0]] = item[1]
            cropStoreFile.create_dataset(name='Affine matrix', data=self.H)


class TemperatureStabilizer(QtCore.QObject):

    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.setPoint = main.tempSetPoint
        self.main.andor.temperature_setpoint = self.setPoint
        self.stableText = 'Temperature has stabilized at set point.'

    def start(self):
        self.main.andor.cooler_on = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10000)
        self.update()

    def stop(self):
        self.timer.stop()

    def update(self):
        tempStatus = self.main.andor.temperature_status
        self.main.tempStatus.setText(tempStatus)
        temperature = np.round(self.main.andor.temperature, 1)
        self.main.temp.setText('{} ºC'.format(temperature))

        if tempStatus != self.stableText:
            threshold = 0.8 * self.setPoint
            if temperature <= threshold or self.main.andor.mock:
                self.main.enableLiveview()

        else:
            self.timer.stop()


class TormentaGUI(QtGui.QMainWindow):

    liveviewStarts = QtCore.pyqtSignal()
    liveviewEnds = QtCore.pyqtSignal()

    def __init__(self, andor, redlaser, bluelaser, greenlaser, scanZ, daq,
                 aptMotor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        lantz.log.log_to_screen(lantz.log.CRITICAL)

        self.andor = andor
        self.shape = self.andor.detector_shape
        self.frameStart = (1, 1)
        self.redlaser = redlaser
        self.greenlaser = greenlaser
        self.bluelaser = bluelaser
        self.scanZ = scanZ
        self.daq = daq
        self.aptMotor = aptMotor

        self.s = Q_(1, 's')
        self.lastTime = ptime.time()
        self.fps = None

        self.umxpx = 0.133

        # Actions in menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        self.savePresetAction = QtGui.QAction('Save configuration...', self)
        self.savePresetAction.setStatusTip('Save camera & recording settings')

        def savePresetFunction():
            return guitools.savePreset(self)
        self.savePresetAction.triggered.connect(savePresetFunction)
        fileMenu.addAction(self.savePresetAction)

        fileMenu.addSeparator()

        self.exportTiffAction = QtGui.QAction('Export HDF5 to Tiff...', self)
        self.exportTiffAction.setStatusTip('Export HDF5 file to Tiff format')
        self.exportTiffAction.triggered.connect(pyqtsub.TiffConverterThread)
        fileMenu.addAction(self.exportTiffAction)

        def tiff2pngFunction():
            return guitools.tiff2png(self)
        self.tiff2pngAction = QtGui.QAction('Export TIFF snaps to PNG...',
                                            self)
        self.tiff2pngAction.setStatusTip('Export HDF5 file to Tiff format')
        self.tiff2pngAction.triggered.connect(tiff2pngFunction)
        fileMenu.addAction(self.tiff2pngAction)

        self.exportlastAction = QtGui.QAction('Export last recording to Tiff',
                                              self)
        self.exportlastAction.setEnabled(False)
        self.exportlastAction.setShortcut('Ctrl+L')
        self.exportlastAction.setStatusTip('Export last recording to Tiff ' +
                                           'format')
        fileMenu.addAction(self.exportlastAction)

        fileMenu.addSeparator()

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.QApplication.closeAllWindows)
        fileMenu.addAction(exitAction)

        optMenu = menubar.addMenu('&After measurement')
        self.flipAfter = QtGui.QAction('Move flipper up', optMenu,
                                       checkable=True)
        optMenu.addAction(self.flipAfter)
        self.flipAfter.setChecked(True)
        self.uvOff = QtGui.QAction('Turn 405nm laser off', optMenu,
                                   checkable=True)
        optMenu.addAction(self.uvOff)
        self.uvOff.setChecked(True)

        # TIRF menu
        tirfMenu = menubar.addMenu('&TIRF')
        self.findTIRFAc = QtGui.QAction('Find TIRF position', tirfMenu)
        self.findTIRFAc.setEnabled(False)
        tirfMenu.addAction(self.findTIRFAc)

        # 3D menu
        threeDMenu = menubar.addMenu('&3D')
        self.threeDAction = QtGui.QAction('Calibrate Z astigmatism',
                                          threeDMenu)
        self.threeDAction.setEnabled(False)
        threeDMenu.addAction(self.threeDAction)
        self.threeDThread = QtCore.QThread(self)
        self.calibrate3D = pyqtsub.Calibrate3D(self)
        self.calibrate3D.moveToThread(self.threeDThread)
        self.threeDAction.triggered.connect(self.threeDThread.start)
        self.calibrate3D.doneSignal.connect(self.threeDThread.quit)
        self.threeDThread.started.connect(self.calibrate3D.start)

        # Analysis menu
        analysisMenu = menubar.addMenu('&Analysis')

        text = 'Affine transform stacks or snaps...'
        self.HtransformAction = QtGui.QAction(text, self)
        tip = ('Correct stacks or single shots using an affine ' +
               'transformation matrix')
        self.HtransformAction.setStatusTip(tip)
        analysisMenu.addAction(self.HtransformAction)
        self.transformerThread = QtCore.QThread(self)
        self.transformer = reg.HtransformStack()
        self.transformer.moveToThread(self.transformerThread)
        self.transformer.finished.connect(self.transformerThread.quit)
        self.transformerThread.started.connect(self.transformer.run)
        self.HtransformAction.triggered.connect(self.transformerThread.start)

        text = 'Subtract background from stacks...'
        self.bkgSubtAction = QtGui.QAction(text, self)
        tip = 'Remove noise from data using a running median filter'
        self.bkgSubtAction.setStatusTip(tip)
        analysisMenu.addAction(self.bkgSubtAction)

        self.bkgSubtThread = QtCore.QThread(self)
        self.bkgSubtractor = stack.BkgSubtractor(self)
        self.bkgSubtractor.moveToThread(self.bkgSubtThread)
        self.bkgSubtractor.finished.connect(self.bkgSubtThread.quit)
        self.bkgSubtThread.started.connect(self.bkgSubtractor.run)
        self.bkgSubtAction.triggered.connect(self.bkgSubtThread.start)

        self.tree = pyqtsub.CamParamTree(self.andor)

        # Frame signals
        frameParam = self.tree.p.param('Field of view')
        frameParam.param('Shape').sigValueChanged.connect(self.updateFrame)
        # Indicator for loading frame shape from a preset setting
        self.customFrameLoaded = False
        self.cropLoaded = False

        # Exposition signals
        def changeExposure():
            return self.changeParameter(self.setExposure)
        timingsPar = self.tree.p.param('Timings')
        self.expPar = timingsPar.param('Set exposure time')
        self.expPar.sigValueChanged.connect(changeExposure)
        self.FTMPar = timingsPar.param('Frame Transfer Mode')
        self.FTMPar.sigValueChanged.connect(changeExposure)
        self.HRRatePar = timingsPar.param('Horizontal readout rate')
        self.HRRatePar.sigValueChanged.connect(changeExposure)
        vertShiftPar = timingsPar.param('Vertical pixel shift')
        self.vertShiftSpeedPar = vertShiftPar.param('Speed')
        self.vertShiftSpeedPar.sigValueChanged.connect(changeExposure)
        self.vertShiftAmpPar = vertShiftPar.param('Clock voltage amplitude')
        self.vertShiftAmpPar.sigValueChanged.connect(changeExposure)
        changeExposure()    # Set default values
        self.cropParam = timingsPar.param('Cropped sensor mode')
        self.cropParam.param('Enable').sigValueChanged.connect(self.cropCCD)

        # Gain signals
        self.PreGainPar = self.tree.p.param('Gain').param('Pre-amp gain')

        def updateGain():
            return self.changeParameter(self.setGain)
        self.PreGainPar.sigValueChanged.connect(updateGain)
        self.GainPar = self.tree.p.param('Gain').param('EM gain')
        self.GainPar.sigValueChanged.connect(updateGain)
        updateGain()        # Set default values

        self.presetsMenu = QtGui.QComboBox()
        presetPath = os.path.join(os.getcwd(), 'tormenta/control/Presets')
        self.presetDir = presetPath
        if not(os.path.isdir(self.presetDir)):
            self.presetDir = presetPath
        try:
            for preset in os.listdir(self.presetDir):
                self.presetsMenu.addItem(preset)
        except FileNotFoundError:
            pass
        self.loadPresetButton = QtGui.QPushButton('Load preset')
        self.loadPresetButton.setFixedWidth(80)

        def loadPresetFunction():
            return guitools.loadPreset(self)
        self.loadPresetButton.pressed.connect(loadPresetFunction)

        # Liveview functionality
        self.liveviewButton = QtGui.QPushButton('LIVEVIEW')
        self.liveviewButton.setStyleSheet("font-size:18px")
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        self.liveviewButton.clicked.connect(self.liveview)
        self.liveviewButton.setEnabled(False)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)
        self.liveShortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+Space'),
                                            self, self.liveviewKey)
        self.liveShortcut.setEnabled(False)

        # viewBox custom Tools
        self.gridButton = QtGui.QPushButton('Grid')
        self.gridButton.setCheckable(True)
        self.gridButton.setEnabled(False)
        self.gridTwoChButton = QtGui.QPushButton('Two-color grid 128px')
        self.gridTwoChButton.setCheckable(True)
        self.gridTwoChButton.setEnabled(False)
        self.gridTwoCh82Button = QtGui.QPushButton('Two-color grid 82px')
        self.gridTwoCh82Button.setCheckable(True)
        self.gridTwoCh82Button.setEnabled(False)
        self.crosshairButton = QtGui.QPushButton('Crosshair')
        self.crosshairButton.setCheckable(True)
        self.crosshairButton.setEnabled(False)

        self.flipperButton = QtGui.QPushButton('x1000')
        self.flipperButton.setStyleSheet("font-size:16px")
        self.flipperButton.setCheckable(True)
        self.flipperButton.clicked.connect(self.daq.toggleFlipper)

        self.viewCtrl = QtGui.QWidget()
        self.viewCtrlLayout = QtGui.QGridLayout()
        self.viewCtrl.setLayout(self.viewCtrlLayout)
        self.viewCtrlLayout.addWidget(self.liveviewButton, 0, 0, 1, 4)
        self.viewCtrlLayout.addWidget(self.gridButton, 1, 0)
        self.viewCtrlLayout.addWidget(self.gridTwoChButton, 1, 1)
        self.viewCtrlLayout.addWidget(self.gridTwoCh82Button, 1, 2)
        self.viewCtrlLayout.addWidget(self.crosshairButton, 1, 3)
        self.viewCtrlLayout.addWidget(self.flipperButton, 2, 0, 1, 4)

        # Status bar info
        self.fpsBox = QtGui.QLabel()
        self.fpsBox.setText('0 fps')
        self.statusBar().addPermanentWidget(self.fpsBox)
        self.tempStatus = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.tempStatus)
        self.temp = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.temp)
        self.cursorPos = QtGui.QLabel()
        self.cursorPos.setText('0, 0')
        self.statusBar().addPermanentWidget(self.cursorPos)
        self.cursorPosInt = QtGui.QLabel()
        self.cursorPosInt.setText('0 counts')
        self.statusBar().addPermanentWidget(self.cursorPosInt)

        # Temperature stabilization functionality
        self.tempSetPoint = -50     # in degC
        self.stabilizer = TemperatureStabilizer(self)
        self.stabilizerThread = QtCore.QThread(self)
        self.stabilizer.moveToThread(self.stabilizerThread)
        self.stabilizerThread.started.connect(self.stabilizer.start)
        self.stabilizerThread.start()
        self.liveviewStarts.connect(self.stabilizer.stop)
        self.liveviewEnds.connect(self.stabilizer.start)

        # Recording settings widget
        self.recWidget = RecordingWidget(self)
        loadMatrixParam = self.tree.fovGroup.param('Load matrix')
        loadMatrixParam.sigActivated.connect(self.recWidget.loadH)
        expTime = self.tree.timeParams.param('Set exposure time')
        expTime.sigValueChanged.connect(self.recWidget.updateRemaining)
        self.snapShortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+S'), self,
                                            self.recWidget.snap)
        self.snapShortcut.setEnabled(False)
        self.recShortcut = QtGui.QShortcut(QtGui.QKeySequence('Ctrl+D'), self,
                                           self.recWidget.startRecording)
        self.recShortcut.setEnabled(False)

        # Hide button
        self.hideColumnButton = QtGui.QPushButton()
        self.hideColumnButton.setFixedWidth(10)
        self.hideColumnButton.setFixedHeight(60)
        self.hideColumnButton.setCheckable(True)

        def hideColumn():
            guitools.hideColumn(self)
        self.hideColumnButton.clicked.connect(hideColumn)

        # Image Widget
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=1, col=1)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.lut = viewbox_tools.cubehelix()
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.vb.setLimits(yMin=0, yMax=20000)
        cubehelix = viewbox_tools.cubehelix().astype(int)
        pos, color = np.arange(0, 1, 1/256), cubehelix
        self.hist.gradient.setColorMap(pg.ColorMap(pos, color))
        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=1, col=2)

        self.grid = viewbox_tools.Grid(self.vb, self.shape)
        self.gridButton.clicked.connect(self.grid.toggle)
        self.gridTwoCh = viewbox_tools.TwoColorGrid(self.vb, 128)
        self.gridTwoChButton.clicked.connect(self.gridTwoCh.toggle)
        self.gridTwoCh82 = viewbox_tools.TwoColorGrid(self.vb, 82)
        self.gridTwoCh82Button.clicked.connect(self.gridTwoCh82.toggle)
        self.crosshair = viewbox_tools.Crosshair(self.vb)
        self.crosshairButton.clicked.connect(self.crosshair.toggle)

        # x and y profiles
        xPlot = imageWidget.addPlot(row=0, col=1)
        xPlot.hideAxis('left')
        xPlot.hideAxis('bottom')
        self.xProfile = xPlot.plot()
        imageWidget.ci.layout.setRowMaximumHeight(0, 40)
        xPlot.setXLink(self.vb)
        yPlot = imageWidget.addPlot(row=1, col=0)
        yPlot.hideAxis('left')
        yPlot.hideAxis('bottom')
        self.yProfile = yPlot.plot()
        self.yProfile.rotate(90)
        imageWidget.ci.layout.setColumnMaximumWidth(0, 40)
        yPlot.setYLink(self.vb)

        # Initial camera configuration taken from the parameter tree
        self.andor.set_exposure_time(self.expPar.value() * self.s)
        self.adjustFrame()

        # Dock widget
        dockArea = DockArea()

        # Console widget
        consoleDock = Dock("Console", size=(600, 200))
        console = ConsoleWidget(namespace={'pg': pg, 'np': np})
        consoleDock.addWidget(console)
        dockArea.addDock(consoleDock)

        # Emission filters table widget
        wheelDock = Dock("Emission filters", size=(20, 20))
        wheelDock.addWidget(FilterTable(editable=True, sortable=False))
        dockArea.addDock(wheelDock, 'top', consoleDock)

        # On time widget
        ontimeDock = Dock('On time histogram', size=(1, 1))
        self.ontimeWidget = ontime.OntimeWidget()
        ontimeDock.addWidget(self.ontimeWidget)
        dockArea.addDock(ontimeDock, 'above', wheelDock)

        # Molecule counting widget
        moleculesDock = Dock('Molecule counting', size=(1, 1))
        self.moleculeWidget = moleculesCounter.MoleculeWidget()
        moleculesDock.addWidget(self.moleculeWidget)
        dockArea.addDock(moleculesDock, 'above', ontimeDock)

        # Focus lock widget
        focusDock = Dock("Focus Control", size=(1, 1))
        # TODO: sacar self.daq
        self.focusWidget = focus.FocusWidget(scanZ, self.recWidget)
        focusDock.addWidget(self.focusWidget)
        dockArea.addDock(focusDock, 'top', consoleDock)

        laserDock = Dock("Laser Control", size=(1, 1))
        self.lasers = (bluelaser, greenlaser, redlaser)
        self.laserWidgets = lasercontrol.LaserWidget(self, self.lasers,
                                                     self.daq, self.aptMotor)
        laserDock.addWidget(self.laserWidgets)
        dockArea.addDock(laserDock, 'above', moleculesDock)
        self.findTIRFAc.triggered.connect(self.laserWidgets.moveMotor.findTIRF)

        # Camera settings widget
        self.cameraWidget = QtGui.QFrame()
        self.cameraWidget.setFrameStyle(QtGui.QFrame.Panel |
                                        QtGui.QFrame.Raised)
        cameraTitle = QtGui.QLabel('<h2><strong>Camera settings</strong></h2>')
        cameraTitle.setTextFormat(QtCore.Qt.RichText)
        cameraGrid = QtGui.QGridLayout()
        self.cameraWidget.setLayout(cameraGrid)
        cameraGrid.addWidget(cameraTitle, 0, 0)
        cameraGrid.addWidget(self.presetsMenu, 1, 0)
        cameraGrid.addWidget(self.loadPresetButton, 1, 1)
        cameraGrid.addWidget(self.tree, 2, 0, 1, 2)

        self.setWindowTitle('Tormenta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Widgets' layout
        self.layout = QtGui.QGridLayout()
        self.cwidget.setLayout(self.layout)
        self.layout.setColumnMinimumWidth(0, 350)
        self.layout.setColumnMinimumWidth(3, 1000)
        self.layout.setRowMinimumHeight(1, 720)
        self.layout.setRowMinimumHeight(2, 40)
        self.layout.setRowMinimumHeight(3, 30)
        self.layout.addWidget(self.cameraWidget, 0, 0, 4, 2)
        self.layout.addWidget(self.viewCtrl, 4, 0, 1, 2)
        self.layout.addWidget(self.recWidget, 5, 0, 1, 2)
        self.layout.addWidget(self.hideColumnButton, 0, 2, 5, 1)
        self.layout.addWidget(imageWidget, 0, 3, 6, 1)
        self.layout.addWidget(dockArea, 0, 4, 6, 1)

        self.showMaximized()

    def flipperInPath(self, value):
        self.flipperButton.setChecked(not(value))
        self.daq.flipper = value

    def cropCCD(self):

        if self.cropParam.param('Enable').value():

            # Used when cropmode is loaded from a config file
            if self.cropLoaded:
                self.startCropMode()

            else:
                self.FTMPar.setWritable()
                if self.shape != self.andor.detector_shape:
                    self.shape = self.andor.detector_shape
                    self.frameStart = (1, 1)
                    self.changeParameter(self.adjustFrame)

                ROIpos = (0, 0)
                self.cropROI = viewbox_tools.ROI(self.shape, self.vb, ROIpos,
                                                 handlePos=(1, 1),
                                                 movable=False,
                                                 handleCenter=(0, 0),
                                                 scaleSnap=True,
                                                 translateSnap=True)
                # Signals
                applyParam = self.cropParam.param('Apply')
                applyParam.sigStateChanged.connect(self.startCropMode)

        else:
            self.cropROI.hide()
            self.shape = self.andor.detector_shape
            self.changeParameter(lambda: self.setCropMode(False))

    def startCropMode(self):

        # Used when cropmode is loaded from a config file
        ROISize = self.cropROI.size()
        self.shape = (int(ROISize[0]), int(ROISize[1]))
        self.cropROI.hide()

        self.frameStart = (1, 1)
        self.andor.crop_mode_shape = self.shape
        self.changeParameter(lambda: self.setCropMode(True))
        self.vb.setLimits(xMin=-0.5, xMax=self.shape[0] - 0.5, minXRange=4,
                          yMin=-0.5, yMax=self.shape[1] - 0.5, minYRange=4)
        self.updateTimings()

        self.grid.update(self.shape)
        self.updateLevels(self.image)     # not working  # TODO: make this work

    def setCropMode(self, state):
        self.andor.crop_mode = state
        self.tree.cropModeParam.param('Apply').hide()
        if not(state):
            self.shape = self.andor.detector_shape
            self.adjustFrame()

    def changeParameter(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        status = self.andor.status
        if status != ('Camera is idle, waiting for instructions.'):
            self.viewtimer.stop()
            self.andor.abort_acquisition()

        function()

        if status != ('Camera is idle, waiting for instructions.'):
            self.andor.start_acquisition()
            time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))
            # 35ms ~ video-rate
            self.viewtimer.start(20)

    def updateLevels(self, image):
        cmin, cmax = guitools.bestLimits(image)
        self.hist.setLevels(cmin, cmax)
#        std = np.std(image)
#        self.hist.setLevels(np.min(image) - std, np.max(image) + std)

    def setGain(self):
        """ Method to change the pre-amp gain and main gain of the EMCCD
        """
        PreAmpGain = self.PreGainPar.value()
        n = np.where(self.andor.PreAmps == PreAmpGain)[0][0]
        # The (2 - n) accounts for the difference in order between the options
        # in the GUI and the camera settings
        self.andor.preamp = 2 - n
        self.andor.EM_gain = self.GainPar.value()

    def setExposure(self):
        """ Method to change the exposure time setting
        """
        self.andor.set_exposure_time(self.expPar.value() * self.s)
        self.andor.frame_transfer_mode = self.FTMPar.value()
        hhRatesArr = np.array([item.magnitude for item in self.andor.HRRates])
        n_hrr = np.where(hhRatesArr == self.HRRatePar.value().magnitude)[0][0]
        # The (3 - n) accounts for the difference in order between the options
        # in the GUI and the camera settings
        self.andor.horiz_shift_speed = 3 - n_hrr

        magList = [item.magnitude for item in self.andor.vertSpeeds]
        n_vss = np.where(np.array(magList) ==
                         self.vertShiftSpeedPar.value().magnitude)[0][0]
        self.andor.vert_shift_speed = n_vss

        n_vsa = np.where(np.array(self.andor.vertAmps) ==
                         self.vertShiftAmpPar.value())[0][0]
        self.andor.set_vert_clock(n_vsa)

        self.updateTimings()

    def adjustFrame(self):
        """ Method to change the area of the CCD to be used and adjust the
        image widget accordingly. It needs a previous change in self.shape
        and self.frameStart)
        """
        self.andor.set_image(shape=self.shape, p_0=self.frameStart)
        self.vb.setLimits(xMin=-0.5, xMax=self.shape[0] - 0.5, minXRange=4,
                          yMin=-0.5, yMax=self.shape[1] - 0.5, minYRange=4)

        self.updateTimings()

        self.grid.update(self.shape)
        self.recWidget.shape = self.shape
        self.recWidget.nChanged()

    def fullChip(self):
        try:
            self.ROI.hide()
        except:
            pass
        self.shape = self.andor.detector_shape
        self.frameStart = (1, 1)
        self.changeParameter(self.adjustFrame)
        self.gridTwoCh.setDimensions()
        self.gridTwoCh82.setDimensions()

    def updateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        frameParam = self.tree.p.param('Field of view')
        shapeStr = frameParam.param('Shape').value()
        if shapeStr == 'Custom':

            if self.shape != self.andor.detector_shape:
                self.fullChip()

            if not(self.customFrameLoaded):
                ROIpos = (0.5 * self.shape[0] - 64, 0.5 * self.shape[1] - 64)
                self.ROI = viewbox_tools.ROI(self.shape, self.vb, ROIpos,
                                             handlePos=(1, 0),
                                             handleCenter=(0, 1),
                                             scaleSnap=True,
                                             translateSnap=True)
                # Signals
                applyParam = frameParam.param('Apply')
                applyParam.sigActivated.connect(self.customFrame)

        elif shapeStr == 'Full chip':
            self.fullChip()

        elif shapeStr.startswith('Two-colors'):
            side = int(shapeStr.split()[1][:-2])
            self.shape = (side*2 + 10, side*2 + 10)
            self.frameStart = (256 - side - 5, int(0.5*(512 - (side*3 + 20))))
            self.changeParameter(self.adjustFrame)
            self.gridTwoCh.changeToSmall(side)
            self.gridTwoCh82.changeToSmall(side)

        else:
            try:
                self.ROI.hide()
            except:
                pass
            side = int(frameParam.param('Shape').value().split('x')[0])
            self.shape = (side, side)
            start = int(0.5*(self.andor.detector_shape[0] - side))
            self.frameStart = (start, start)

            self.changeParameter(self.adjustFrame)

        self.recWidget.nChanged()

    def customFrame(self):

        ROISize = self.ROI.size()
        self.shape = (int(ROISize[0]), int(ROISize[0]))
        self.frameStart = (int(self.ROI.pos()[0]), int(self.ROI.pos()[0]))

        self.changeParameter(self.adjustFrame)
        self.ROI.hide()
        self.grid.update(self.shape)
        self.recWidget.shape = self.shape
        self.tree.fovGroup.param('Apply').hide()

    def updateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree.
        """
        timings = self.andor.acquisition_timings
        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings
        timingsPar = self.tree.p.param('Timings')
        RealExpPar = timingsPar.param('Real exposure time')
        RealAccPar = timingsPar.param('Real accumulation time')
        EffFRPar = timingsPar.param('Effective frame rate')
        RealExpPar.setValue(self.t_exp_real.magnitude)
        RealAccPar.setValue(self.t_acc_real.magnitude)
        EffFRPar.setValue(1 / self.t_acc_real.magnitude)

    def enableLiveview(self):
        self.liveviewButton.setEnabled(True)
        self.liveShortcut.setEnabled(True)

    # This is the function triggered by the liveview shortcut
    def liveviewKey(self):

        if self.liveviewButton.isChecked():
            self.liveviewStop()
            self.liveviewButton.setChecked(False)

        else:
            self.liveviewStart(True)
            self.liveviewButton.setChecked(True)

    # This is the function triggered by pressing the liveview button
    def liveview(self, update=True):
        """ Image live view when not recording
        """
        if self.liveviewButton.isChecked():
            self.liveviewStart(update)

        else:
            self.liveviewStop()

    def mouseMoved(self, pos):
        guitools.mouseMoved(self, pos)

    def liveviewStart(self, update):

        self.liveviewStarts.emit()

        idle = 'Camera is idle, waiting for instructions.'
        if self.andor.status != idle:
            self.andor.abort_acquisition()

        self.andor.acquisition_mode = 'Run till abort'
        self.andor.shutter(0, 1, 0, 0, 0)

        self.andor.start_acquisition()
        time.sleep(np.max((5 * self.t_exp_real.magnitude, 1)))
        self.recWidget.readyToRecord = True
        self.recWidget.recButton.setEnabled(True)

        # Initial image
        self.image = np.transpose(self.andor.most_recent_image16(self.shape))
        self.img.setImage(self.image, autoLevels=False)
        if update:
            self.updateLevels(self.image)
        self.viewtimer.start(20)
        self.moleculeWidget.enableBox.setEnabled(True)
        self.gridButton.setEnabled(True)
        self.gridTwoChButton.setEnabled(True)
        self.gridTwoCh82Button.setEnabled(True)
        self.crosshairButton.setEnabled(True)
        self.snapShortcut.setEnabled(True)
        self.findTIRFAc.setEnabled(True)
        self.threeDAction.setEnabled(True)

        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)

    def liveviewStop(self):
        self.findTIRFAc.setEnabled(False)
        self.threeDAction.setEnabled(False)
        self.snapShortcut.setEnabled(False)
        self.viewtimer.stop()
        self.recWidget.readyToRecord = False
        self.moleculeWidget.enableBox.setEnabled(False)
        self.gridButton.setChecked(False)
        self.gridButton.setEnabled(False)
        self.grid.hide()
        self.gridTwoChButton.setChecked(False)
        self.gridTwoChButton.setEnabled(False)
        self.gridTwoCh.hide()
        self.gridTwoCh82Button.setChecked(False)
        self.gridTwoCh82Button.setEnabled(False)
        self.gridTwoCh82.hide()
        self.crosshairButton.setChecked(False)
        self.crosshairButton.setEnabled(False)
        self.crosshair.hide()

        # Turn off camera, close shutter
        idleMsg = 'Camera is idle, waiting for instructions.'
        if self.andor.status != idleMsg:
            self.andor.abort_acquisition()

        self.andor.shutter(0, 2, 0, 0, 0)
        self.img.setImage(np.zeros(self.shape), autoLevels=False)

        self.liveviewEnds.emit()

    def updateView(self):
        """ Image update while in Liveview mode
        """
        try:
            newData = self.andor.most_recent_image16(self.shape)
            self.image = np.transpose(newData)
            if self.moleculeWidget.enabled:
                self.moleculeWidget.graph.update(self.image)
            self.img.setImage(self.image, autoLevels=False)

            if self.crosshair.showed:
                ycoord = int(np.round(self.crosshair.hLine.pos()[1]))
                xcoord = int(np.round(self.crosshair.vLine.pos()[0]))
                self.xProfile.setData(self.image[:, ycoord])
                self.yProfile.setData(self.image[xcoord])

            self.fpsMath()

        except:
            pass

    def fpsMath(self):
        now = ptime.time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt * 3., 0, 1)
            self.fps = self.fps * (1 - s) + (1.0/dt) * s
        self.fpsBox.setText('{} fps'.format(int(self.fps)))

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
        self.viewtimer.stop()
        self.stabilizer.timer.stop()
        self.stabilizerThread.terminate()

        # Turn off camera, close shutter and flipper
        if self.andor.status != 'Camera is idle, waiting for instructions.':
            self.andor.abort_acquisition()
        self.andor.shutter(0, 2, 0, 0, 0)
        self.daq.flipper = True

        self.laserWidgets.closeEvent(*args, **kwargs)
        self.focusWidget.closeEvent(*args, **kwargs)
        super().closeEvent(*args, **kwargs)
