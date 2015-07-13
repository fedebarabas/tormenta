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

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.dockarea import Dock, DockArea
from pyqtgraph.console import ConsoleWidget

from tkinter import Tk, filedialog, messagebox
import h5py as hdf
import tifffile as tiff     # http://www.lfd.uci.edu/~gohlke/pythonlibs/#vlfd
from lantz import Q_

# tormenta imports
import tormenta.control.instruments as instruments
import tormenta.control.lasercontrol as lasercontrol
import tormenta.control.focus as focus
import tormenta.control.molecules_counter as moleculesCounter
import tormenta.control.ontime as ontime
import tormenta.control.guitools as guitools


class RecordingWidget(QtGui.QFrame):

    def __init__(self, main, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.dataname = 'data'      # In case I need a QLineEdit for this
        self.shape = self.main.shape
        self.initialDir = r'C:\Users\Usuario\Documents\Data'

        # Title
        recTitle = QtGui.QLabel('<h2><strong>Recording</strong></h2>')
        recTitle.setTextFormat(QtCore.Qt.RichText)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Folder and filename fields
        self.folderEdit = QtGui.QLineEdit(self.initialDir)
        openFolderButton = QtGui.QPushButton('Open')
        openFolderButton.clicked.connect(self.openFolder)
        loadFolderButton = QtGui.QPushButton('Load...')
        loadFolderButton.clicked.connect(self.loadFolder)
        self.filenameEdit = QtGui.QLineEdit('filename')

        # Snap and recording buttons
        self.snapTIFFButton = QtGui.QPushButton('Snap TIFF')
        self.snapTIFFButton.setStyleSheet("font-size:16px")
        self.snapTIFFButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        self.snapTIFFButton.clicked.connect(self.snapTIFF)
        self.snapHDFButton = QtGui.QPushButton('Snap HDF5')
        self.snapHDFButton.setStyleSheet("font-size:16px")
        self.snapHDFButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                         QtGui.QSizePolicy.Expanding)
        self.snapHDFButton.clicked.connect(self.snapHDF)
        self.recButton = QtGui.QPushButton('REC')
        self.recButton.setStyleSheet("font-size:16px")
        self.recButton.setCheckable(True)
        self.recButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                     QtGui.QSizePolicy.Expanding)
        self.recButton.clicked.connect(self.startRecording)

        # Number of frames and measurement timing
        self.currentFrame = QtGui.QLabel('0 /')
        self.currentFrame.setAlignment((QtCore.Qt.AlignRight |
                                        QtCore.Qt.AlignVCenter))
        self.currentFrame.setFixedWidth(45)
        self.numExpositionsEdit = QtGui.QLineEdit('100')
        self.numExpositionsEdit.setFixedWidth(45)
        zeroTime = datetime.timedelta(seconds=0)
        self.tElapsed = QtGui.QLabel('Elapsed: {}'.format(zeroTime))
        self.tRemaining = QtGui.QLabel()
        self.tRemaining.setAlignment((QtCore.Qt.AlignRight |
                                      QtCore.Qt.AlignVCenter))
        self.numExpositionsEdit.textChanged.connect(self.nChanged)
        self.updateRemaining()

        # Layout
        buttonWidget = QtGui.QWidget()
        buttonGrid = QtGui.QGridLayout()
        buttonWidget.setLayout(buttonGrid)
        buttonGrid.addWidget(self.snapTIFFButton, 0, 0)
        buttonGrid.addWidget(self.snapHDFButton, 0, 1)
        buttonGrid.addWidget(self.recButton, 0, 2)

        recGrid = QtGui.QGridLayout()
        self.setLayout(recGrid)

        recGrid.addWidget(recTitle, 0, 0, 1, 3)
        recGrid.addWidget(QtGui.QLabel('Folder'), 1, 0)
        recGrid.addWidget(loadFolderButton, 1, 5)
        recGrid.addWidget(openFolderButton, 1, 4)
        recGrid.addWidget(self.folderEdit, 2, 0, 1, 6)
        recGrid.addWidget(QtGui.QLabel('Filename'), 3, 0, 1, 2)
        recGrid.addWidget(self.filenameEdit, 4, 0, 1, 6)
        recGrid.addWidget(QtGui.QLabel('Number of expositions'), 5, 0)
        recGrid.addWidget(self.currentFrame, 5, 1)
        recGrid.addWidget(self.numExpositionsEdit, 5, 2)
        recGrid.addWidget(self.tElapsed, 5, 4)
        recGrid.addWidget(self.tRemaining, 5, 5)
        recGrid.addWidget(buttonWidget, 6, 0, 1, 6)

        recGrid.setColumnMinimumWidth(0, 110)
        recGrid.setColumnMinimumWidth(3, 20)
        recGrid.setRowMinimumHeight(6, 60)

        self.writable = True
        self.readyToRecord = False

    @property
    def readyToRecord(self):
        return self._readyToRecord

    @readyToRecord.setter
    def readyToRecord(self, value):
        self.snapTIFFButton.setEnabled(value)
        self.snapHDFButton.setEnabled(value)
        self.recButton.setEnabled(value)
        self._readyToRecord = value

    @property
    def writable(self):
        return self._writable

    @writable.setter
    def writable(self, value):
        self.folderEdit.setEnabled(value)
        self.filenameEdit.setEnabled(value)
        self.numExpositionsEdit.setEnabled(value)
        self._writable = value

    def n(self):
        text = self.numExpositionsEdit.text()
        if text == '':
            return 0
        else:
            return int(text)

    def nChanged(self):
        self.updateRemaining()
        self.limitExpositions(9)

    def updateRemaining(self):
        rSecs = self.main.t_acc_real.magnitude * self.n()
        rTime = datetime.timedelta(seconds=np.round(rSecs))
        self.tRemaining.setText('Remaining: {}'.format(rTime))

    def nPixels(self):
        return self.shape[0] * self.shape[1]

    # Setting a xGB limit on file sizes to be able to open them in Fiji
    def limitExpositions(self, xGB):
        # nMax = xGB * 8 * 1024**3 / (pixels * 16)
        nMax = xGB * 2**29 / self.nPixels()
        if self.n() > nMax:
            self.numExpositionsEdit.setText(str(np.round(nMax).astype(int)))

    def openFolder(self, path):
        print(self.folderEdit.text())
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
            self.folderEdit.setText(folder)
        except OSError:
            print(self.folderEdit.text())
            pass

    # Attributes saving
    def getAttrs(self):
        attrs = self.main.tree.attrs()
        attrs.extend([('Date', time.strftime("%Y-%m-%d")),
                      ('Start time', time.strftime("%H:%M:%S")),
                      ('element_size_um', (1, 0.143, 0.143)),
                      ('NA', 1.42),
                      ('lambda_em', 670)])
        for laserControl in self.main.laserWidgets.controls:
            name = re.sub('<[^<]+?>', '', laserControl.name.text())
            attrs.append((name, laserControl.laser.power))
        return attrs

    def snapHDF(self):

        folder = self.folderEdit.text()
        if os.path.exists(folder):

            image = self.main.andor.most_recent_image16(self.shape)

            name = os.path.join(folder, self.filenameEdit.text())
            savename = guitools.getUniqueName(name + '_snap.hdf5')
            store_file = hdf.File(savename)
            store_file.create_dataset(name=self.dataname, data=image)
            for item in self.getAttrs():
                if item[1] is not None:
                    store_file[self.dataname].attrs[item[0]] = item[1]
            store_file.close()

        else:
            self.folderWarning()

    def snapTIFF(self):

        folder = self.folderEdit.text()
        if os.path.exists(folder):

            image = self.main.andor.most_recent_image16(self.shape)
            time.sleep(0.01)

            savename = (os.path.join(folder, self.filenameEdit.text()) +
                        '_snap.tiff')
            savename = guitools.getUniqueName(savename)
            tiff.imsave(savename, image, description=self.dataname,
                        software='Tormenta')
            guitools.attrsToTxt(os.path.splitext(savename)[0], self.getAttrs())

        else:
            self.folderWarning()

    def folderWarning(self):
        root = Tk()
        root.withdraw()
        messagebox.showwarning(title='Warning', message="Folder doesn't exist")
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
        eText = 'Elapsed: {}'.format(datetime.timedelta(seconds=eSecs))
        self.tElapsed.setText(eText)
        nframe = self.worker.j
        rFrames = self.n() - nframe
        rSecs = np.round(self.main.t_acc_real.magnitude * rFrames)
        rText = 'Remaining: {}'.format(datetime.timedelta(seconds=rSecs))
        self.tRemaining.setText(rText)
        self.currentFrame.setText(str(nframe) + ' /')

    def startRecording(self):

        if self.recButton.isChecked():

            folder = self.folderEdit.text()
            if os.path.exists(folder):

                self.writable = False
                self.readyToRecord = False
                self.recButton.setEnabled(True)
                self.recButton.setText('STOP')
                self.main.tree.writable = False
                self.main.liveviewButton.setEnabled(False)
                self.main.viewtimer.stop()

                self.savename = (os.path.join(folder, self.filenameEdit.text())
                                 + '.hdf5')
                self.savename = guitools.getUniqueName(self.savename)
                self.startTime = ptime.time()

                shape = (self.n(), self.shape[0], self.shape[1])
                self.worker = RecWorker(self.main.andor, shape,
                                        self.main.t_exp_real, self.savename,
                                        self.dataname, self.getAttrs())
                self.worker.updateSignal.connect(self.updateGUI)
                self.worker.doneSignal.connect(self.endRecording)
                self.recordingThread = QtCore.QThread()
                self.worker.moveToThread(self.recordingThread)
                self.recordingThread.started.connect(self.worker.start)
                self.recordingThread.start()

            else:
                self.folderWarning()
                self.recButton.setChecked(False)

        else:
            self.worker.pressed = False

    def endRecording(self):

        self.recordingThread.terminate()

        if self.main.focusWidget.focusDataBox.isChecked():
            self.main.focusWidget.exportData()
        else:
            self.main.focusWidget.graph.savedDataSignal = []
            self.main.focusWidget.graph.mean = 0
            self.main.focusWidget.n = 1
            self.main.focusWidget.max_dev = 0

        converterFunction = lambda: guitools.TiffConverterThread(self.savename)
        self.main.exportlastAction.triggered.connect(converterFunction)
        self.main.exportlastAction.setEnabled(True)

        self.writable = True
        self.readyToRecord = True
        self.recButton.setText('REC')
        self.recButton.setChecked(False)
        self.main.tree.writable = True
        self.main.liveviewButton.setEnabled(True)
        self.main.liveviewStart(update=False)


class RecWorker(QtCore.QObject):

    updateSignal = QtCore.pyqtSignal(np.ndarray)
    doneSignal = QtCore.pyqtSignal()

    def __init__(self, andor, shape, t_exp, savename, dataname, attrs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.andor = andor
        self.shape = shape
        self.t_exp = t_exp
        self.savename = savename
        self.dataname = dataname
        self.attrs = attrs
        self.pressed = True

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

        self.store_file = hdf.File(self.savename, "w")
        self.store_file.create_dataset(name=self.dataname, shape=self.shape,
                                       maxshape=self.shape, dtype=np.uint16)
        self.dataset = self.store_file[self.dataname]
        self.startTime = ptime.time()

        while self.j < self.shape[0] and self.pressed:

            time.sleep(self.t_exp.magnitude)
            if self.andor.n_images_acquired > self.j:
                i, self.j = self.andor.new_images_index
                newImages = self.andor.images16(i, self.j, (self.shape[1],
                                                            self.shape[2]),
                                                1, self.shape[0])
                self.dataset[i - 1:self.j] = newImages
                self.updateSignal.emit(self.dataset[self.j - 1])

        # Crop dataset if it's stopped before finishing
        if self.j < self.shape[0]:
            self.dataset.resize((self.j, self.shape[1], self.shape[2]))

        # Saving parameters
        for item in self.attrs:
            if item[1] is not None:
                self.dataset.attrs[item[0]] = item[1]

        self.store_file.close()
        self.doneSignal.emit()


class TemperatureStabilizer(QtCore.QObject):

    def __init__(self, main, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.main = main
        self.parameter = self.main.TempPar
        self.setPointPar = self.parameter.param('Set point')
        self.setPointPar.sigValueChanged.connect(self.updateTemp)
        self.currTempPar = self.parameter.param('Current temperature')

    def updateTemp(self):
        self.main.andor.temperature_setpoint = Q_(self.setPointPar.value(),
                                                  'degC')

    def start(self):
        self.updateTemp()
        self.main.andor.cooler_on = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def update(self):
        tempStatus = self.main.andor.temperature_status
        self.parameter.param('Status').setValue(tempStatus)

        stable = 'Temperature has stabilized at set point.'
        if tempStatus != stable:
            temperature = self.main.andor.temperature
            self.currTempPar.setValue(np.round(temperature.magnitude, 1))
            tempSetPoint = self.main.andor.temperature_setpoint.magnitude
            threshold = Q_(0.8 * tempSetPoint, 'degC')
            if temperature <= threshold or self.main.andor.mock:
                self.main.liveviewButton.setEnabled(True)
                self.main.liveviewAction.setEnabled(True)
            time.sleep(10)

        else:
            self.timer.stop()


class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """

    def __init__(self, andor, *args, **kwargs):
        super().__init__(*args, **kwargs)

        vpssTip = ("Faster vertical shift speeds allow for higher maximum \n"
                   "frame rates but reduce the pixel well depth, causing \n"
                   "degraded spatial resolution (smearing) for bright \n"
                   "signals. To improve the charge transfer efficiency the \n"
                   "vertical clock voltage amplitude can be increased at \n"
                   "the expense of a higher clock-induced charge.")

        preampTip = ("Andor recommend using the highest value setting for \n"
                     "most low-light applications")

        EMGainTip = ("A gain of x4-5 the read noise (see spec sheet) is \n"
                     "enough to render this noise source negligible. In \n"
                     "practice, this can always be achieved with EM Gain of \n"
                     "less than x300 (often much less). Pushing gain beyond \n"
                     "300 would give little or no extra SNR benefit and \n"
                     "would only reduce dynamic range.")

        hrrTip = ("Slower readout typically allows lower read noise and \n"
                  "higher available dynamic range, but at the expense of \n"
                  "slower frame rates")

        croppedTip = ("Ensure that no light is falling on the light \n"
                      "sensitive area outside of the defined region. Any \n"
                      "light collected outside the cropped area could \n"
                      "corrupt the images which were acquired in this mode.")

        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': andor.idn.split(',')[0]},
                  {'name': 'Image frame', 'type': 'group', 'children': [
                      {'name': 'Shape', 'type': 'list',
                       'values': ['Full chip', '256x256', '128x128', '64x64',
                                  'Custom']},
                      {'name': 'Apply', 'type': 'action'}]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                      {'name': 'Horizontal readout rate', 'type': 'list',
                       'values': andor.HRRates, 'tip': hrrTip},
                      {'name': 'Vertical pixel shift', 'type': 'group',
                       'children': [
                           {'name': 'Speed', 'type': 'list',
                            'values': andor.vertSpeeds[::-1],
                            'tip': vpssTip},
                           {'name': 'Clock voltage amplitude', 'tip': vpssTip,
                            'type': 'list', 'values': andor.vertAmps}]},
                      {'name': 'Frame Transfer Mode', 'type': 'bool',
                       'value': False},
                      {'name': 'Cropped sensor mode', 'type': 'group',
                       'children': [
                           {'name': 'Enable', 'type': 'bool', 'value': False,
                            'tip': croppedTip},
                           {'name': 'Apply', 'type': 'action'}]},
                      {'name': 'Set exposure time', 'type': 'float',
                       'value': 0.1, 'limits': (0,
                                                andor.max_exposure.magnitude),
                       'siPrefix': True, 'suffix': 's'},
                      {'name': 'Real exposure time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 's'},
                      {'name': 'Real accumulation time', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 's'},
                      {'name': 'Effective frame rate', 'type': 'float',
                       'value': 0, 'readonly': True, 'siPrefix': True,
                       'suffix': 'Hz'}]},
                  {'name': 'Gain', 'type': 'group', 'children': [
                      {'name': 'Pre-amp gain', 'type': 'list',
                       'values': list(andor.PreAmps),
                       'tip': preampTip},
                      {'name': 'EM gain', 'type': 'int', 'value': 1,
                       'limits': (0, andor.EM_gain_range[1]),
                       'tip': EMGainTip}]},
                  {'name': 'Temperature', 'type': 'group', 'children': [
                      {'name': 'Set point', 'type': 'int', 'value': -50,
                       'suffix': 'º', 'limits': (-80, 0)},
                      {'name': 'Current temperature', 'type': 'int',
                       'value': andor.temperature.magnitude, 'suffix': 'ºC',
                       'readonly': True},
                      {'name': 'Status', 'type': 'str', 'readonly': True,
                       'value': andor.temperature_status}]}]

        self.p = Parameter.create(name='params', type='group', children=params)
        self.setParameters(self.p, showTop=False)
        self._writable = True

        self.timeParams = self.p.param('Timings')
        self.cropModeParam = self.timeParams.param('Cropped sensor mode')
        self.cropModeEnableParam = self.cropModeParam.param('Enable')
        self.cropModeEnableParam.setWritable(False)
        self.frameTransferParam = self.timeParams.param('Frame Transfer Mode')
        self.frameTransferParam.sigValueChanged.connect(self.enableCropMode)

    def enableCropMode(self):
        value = self.frameTransferParam.value()
        if value:
            self.cropModeEnableParam.setWritable(True)
        else:
            self.cropModeEnableParam.setValue(False)
            self.cropModeEnableParam.setWritable(False)

    @property
    def writable(self):
        return self._writable

    @writable.setter
    def writable(self, value):
        self._writable = value
        self.p.param('Image frame').param('Shape').setWritable(value)
        self.timeParams.param('Frame Transfer Mode').setWritable(value)
        croppedParam = self.timeParams.param('Cropped sensor mode')
        croppedParam.param('Enable').setWritable(value)
        self.timeParams.param('Horizontal readout rate').setWritable(value)
        self.timeParams.param('Set exposure time').setWritable(value)
        vpsParams = self.timeParams.param('Vertical pixel shift')
        vpsParams.param('Speed').setWritable(True)
        vpsParams.param('Clock voltage amplitude').setWritable(value)
        gainParams = self.p.param('Gain')
        gainParams.param('Pre-amp gain').setWritable(value)
        gainParams.param('EM gain').setWritable(value)

    def attrs(self):
        attrs = []
        for ParName in self.p.getValues():
            Par = self.p.param(str(ParName))
            if not(Par.hasChildren()):
                attrs.append((str(ParName), Par.value()))
            else:
                for sParName in Par.getValues():
                    sPar = Par.param(str(sParName))
                    if sPar.type() != 'action':
                        if not(sPar.hasChildren()):
                            attrs.append((str(sParName), sPar.value()))
                        else:
                            for ssParName in sPar.getValues():
                                ssPar = sPar.param(str(ssParName))
                                attrs.append((str(ssParName), ssPar.value()))
        return attrs


class TormentaGUI(QtGui.QMainWindow):

    def __init__(self, andor, redlaser, bluelaser, greenlaser, scanZ,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.andor = andor
        self.shape = self.andor.detector_shape
        self.frameStart = (1, 1)
        self.redlaser = redlaser
        self.greenlaser = greenlaser
        self.bluelaser = bluelaser
        self.scanZ = scanZ

        self.setWindowTitle('Tormenta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        self.s = Q_(1, 's')
        self.lastTime = ptime.time()
        self.fps = None

        # Actions and menubar
        self.savePresetAction = QtGui.QAction('Save configuration...', self)
        self.savePresetAction.setShortcut('Ctrl+S')
        self.savePresetAction.setStatusTip('Save camera & recording settings')
        savePresetFunction = lambda: guitools.savePreset(self)
        self.savePresetAction.triggered.connect(savePresetFunction)
        self.liveviewAction = QtGui.QAction(self)
        self.liveviewAction.setShortcut('Ctrl+Space')
        self.liveviewAction.triggered.connect(self.liveviewKey)
        self.liveviewAction.setEnabled(False)
        self.exportTiffAction = QtGui.QAction('Export HDF5 to Tiff...', self)
        self.exportTiffAction.setShortcut('Ctrl+E')
        self.exportTiffAction.setStatusTip('Export HDF5 file to Tiff format')
        self.exportTiffAction.triggered.connect(guitools.TiffConverterThread)
        self.exportlastAction = QtGui.QAction('Export last recording to Tiff',
                                              self)
        self.exportlastAction.setEnabled(False)
        self.exportlastAction.setShortcut('Ctrl+L')
        self.exportlastAction.setStatusTip('Export last recording to Tiff ' +
                                           'format')
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.QApplication.closeAllWindows)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(self.savePresetAction)
        fileMenu.addAction(self.exportTiffAction)
        fileMenu.addAction(self.exportlastAction)
        fileMenu.addAction(exitAction)

        self.statusBar()
        self.tree = CamParamTree(self.andor)

        # Frame signals
        frameParam = self.tree.p.param('Image frame')
        frameParam.param('Shape').sigValueChanged.connect(self.updateFrame)
        # Indicator for loading frame shape from a preset setting
        self.customFrameLoaded = False
        self.cropLoaded = False

        # Exposition signals
        changeExposure = lambda: self.changeParameter(self.setExposure)
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
        updateGain = lambda: self.changeParameter(self.setGain)
        self.PreGainPar.sigValueChanged.connect(updateGain)
        self.GainPar = self.tree.p.param('Gain').param('EM gain')
        self.GainPar.sigValueChanged.connect(updateGain)
        updateGain()        # Set default values

        # Camera settings widget
        cameraWidget = QtGui.QFrame()
        cameraWidget.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        cameraTitle = QtGui.QLabel('<h2><strong>Camera settings</strong></h2>')
        cameraTitle.setTextFormat(QtCore.Qt.RichText)
        cameraGrid = QtGui.QGridLayout()
        cameraWidget.setLayout(cameraGrid)
        cameraGrid.addWidget(cameraTitle, 0, 0)
        cameraGrid.addWidget(self.tree, 1, 0)

        # Temperature stabilization functionality
        self.TempPar = self.tree.p.param('Temperature')
        self.stabilizer = TemperatureStabilizer(self)
        self.stabilizerThread = QtCore.QThread()
        self.stabilizer.moveToThread(self.stabilizerThread)
        self.stabilizerThread.started.connect(self.stabilizer.start)
        self.stabilizerThread.start()

        # Liveview functionality
        self.liveviewButton = QtGui.QPushButton('LIVEVIEW')
        self.liveviewButton.setStyleSheet("font-size:20px")
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                          QtGui.QSizePolicy.Expanding)
        self.liveviewButton.clicked.connect(self.liveview)
        self.liveviewButton.setEnabled(False)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)

        self.presetsMenu = QtGui.QComboBox()
        self.presetDir = r'C:\Users\Usuario\Documents\Data\Presets'
        if not(os.path.isdir(self.presetDir)):
            self.presetDir = os.path.join(os.getcwd(), 'Presets')
        for preset in os.listdir(self.presetDir):
            self.presetsMenu.addItem(preset)
        self.loadPresetButton = QtGui.QPushButton('Load preset')
        loadPresetFunction = lambda: guitools.loadPreset(self)
        self.loadPresetButton.pressed.connect(loadPresetFunction)

        # Recording settings widget
        self.recWidget = RecordingWidget(self)

        # Image Widget
        self.fpsBox = QtGui.QLabel()
        imageWidget = pg.GraphicsLayoutWidget()
        self.vb = imageWidget.addViewBox(row=1, col=1)
        self.vb.setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.lut = guitools.cubehelix()
        self.img.setLookupTable(self.lut)
        self.img.translate(-0.5, -0.5)
        self.vb.addItem(self.img)
        self.vb.setAspectLocked(True)
        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.vb.setLimits(yMin=0, yMax=20000)
        imageWidget.addItem(self.hist, row=1, col=2)

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

        # viewBox custom Tools
        self.gridButton = QtGui.QPushButton('Grid')
        self.gridButton.setCheckable(True)
        self.grid = guitools.Grid(self.vb, self.shape)
        self.gridButton.clicked.connect(self.grid.toggle)
        self.crosshairButton = QtGui.QPushButton('Crosshair')
        self.crosshairButton.setCheckable(True)
        self.crosshair = guitools.Crosshair(self.vb)
        self.crosshairButton.clicked.connect(self.crosshair.toggle)

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
        tableWidget = pg.TableWidget(sortable=False)
        filters = [('ZET642NF',    'Notch 642nm',     4, ''),
                   ('ET700/75m',   'Bandpass 700/75', 5, 'Alexa647, Atto655'),
                   ('FF01-593/40', 'Bandpass 593/40', 6, 'Alexa568, Atto565, '
                                                         'Nile Red'),
                   ('ET575/50',    'Bandpass 575/50', 1, 'Alexa532, Atto550'),
                   ('FF03-525/50', 'Bandpass 525/50', 2, 'GFP'),
                   ('',            '',                3, '')]
        data = np.array(filters, dtype=[('Filtro', object),
                                        ('Descripción', object),
                                        ('Antiposición', int),
                                        ('Fluoróforos', object)])
        tableWidget.setData(data)
        wheelDock.addWidget(tableWidget)
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
#        self.focusWidget = FocusWidget(DAQ, scanZ, self.recWidget)
        self.focusWidget = focus.FocusWidget(scanZ, self.recWidget)
#        self.focusThread = QtCore.QThread()
#        self.focusWidget.moveToThread(self.focusThread)
#        self.focusThread.started.connect(self.focusWidget)
#        self.focusThread.start()
        focusDock.addWidget(self.focusWidget)
        dockArea.addDock(focusDock, 'above', moleculesDock)

        laserDock = Dock("Laser Control", size=(1, 1))
        self.lasers = (redlaser, bluelaser, greenlaser)
        self.laserWidgets = lasercontrol.LaserWidget(self.lasers)
        laserDock.addWidget(self.laserWidgets)
        dockArea.addDock(laserDock, 'above', focusDock)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.setColumnMinimumWidth(0, 380)
        layout.setColumnMinimumWidth(2, 600)
        layout.setColumnMinimumWidth(3, 200)
        layout.setRowMinimumHeight(1, 720)
        layout.setRowMinimumHeight(2, 910)
        layout.setRowMinimumHeight(3, 30)
        layout.addWidget(self.presetsMenu, 0, 0)
        layout.addWidget(self.loadPresetButton, 0, 1)
        layout.addWidget(cameraWidget, 1, 0, 2, 2)
        layout.addWidget(self.liveviewButton, 3, 0, 1, 2)
        layout.addWidget(self.recWidget, 4, 0, 2, 2)
        layout.addWidget(imageWidget, 0, 2, 5, 4)
        layout.addWidget(self.fpsBox, 5, 2)
        layout.addWidget(self.gridButton, 5, 4)
        layout.addWidget(self.crosshairButton, 5, 5)
        layout.addWidget(dockArea, 0, 6, 6, 1)

        layout.setRowMinimumHeight(2, 40)

    def cropCCD(self):

        if self.cropParam.param('Enable').value():

            # Used when cropmode is loaded from a config file
            if self.cropLoaded:
                self.startCropMode()

            else:
                self.FTMPar.setWritable()
                if self.shape != self.andor.detector_shape:
                    self.shape = self.andor.detector_shape
                    self.changeParameter(self.adjustFrame)

                ROIpos = (0, 0)
                self.cropROI = guitools.ROI(self.shape, self.vb, ROIpos,
                                            handlePos=(1, 1), movable=False,
                                            handleCenter=(0, 0),
                                            scaleSnap=True, translateSnap=True)
                # Signals
                applyParam = self.cropParam.param('Apply')
                applyParam.sigStateChanged.connect(self.startCropMode)

        else:
            self.cropROI.hide()
            self.shape = self.andor.detector_shape
            self.changeParameter(lambda: self.setCropMode(False))

    def startCropMode(self):

        # Used when cropmode is loaded from a config file
        if self.cropLoaded:
            ROISize = self.cropROI.size()
            self.shape = (int(ROISize[0]), int(ROISize[1]))
            self.cropROI.hide()

        self.frameStart = (1, 1)
        self.andor.crop_mode_shape = self.shape
        self.changeParameter(lambda: self.setCropMode(True))
        self.vb.setLimits(xMin=-0.5, xMax=self.shape[0] - 0.5,
                          yMin=-0.5, yMax=self.shape[1] - 0.5,
                          minXRange=4, minYRange=4)
        self.updateTimings()

        self.grid.update(self.shape)

    def setCropMode(self, state):
        self.andor.crop_mode = state
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
            self.viewtimer.start(0)

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

        n_vss = np.where(np.array([item.magnitude
                                  for item in self.andor.vertSpeeds])
                         == self.vertShiftSpeedPar.value().magnitude)[0][0]
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

    def updateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        frameParam = self.tree.p.param('Image frame')
        if frameParam.param('Shape').value() == 'Custom':

            if not(self.customFrameLoaded):
                ROIpos = (0.5 * self.shape[0] - 64, 0.5 * self.shape[1] - 64)
                self.ROI = guitools.ROI(self.shape, self.vb, ROIpos,
                                        handlePos=(1, 0), handleCenter=(0, 1),
                                        scaleSnap=True, translateSnap=True)
                # Signals
                applyParam = frameParam.param('Apply')
                applyParam.sigStateChanged.connect(self.customFrame)

        elif frameParam.param('Shape').value() == 'Full chip':
            self.shape = self.andor.detector_shape
            self.frameStart = (1, 1)
            self.changeParameter(self.adjustFrame)

        else:
            side = int(frameParam.param('Shape').value().split('x')[0])
            self.shape = (side, side)
            start = int(0.5*(self.andor.detector_shape[0] - side) + 1)
            self.frameStart = (start, start)

            self.changeParameter(self.adjustFrame)

    def customFrame(self):

        ROISize = self.ROI.size()
        self.shape = (int(ROISize[1]), int(ROISize[0]))
        self.frameStart = (int(self.ROI.pos()[1]), int(self.ROI.pos()[0]))

        self.changeParameter(self.adjustFrame)
        self.ROI.hide()
        self.grid.update(self.shape)
        self.recWidget.shape = self.shape

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

    def liveviewStart(self, update):
        self.stabilizer.timer.stop()
        idle = 'Camera is idle, waiting for instructions.'
        if self.andor.status != idle:
            self.andor.abort_acquisition()

        self.andor.acquisition_mode = 'Run till abort'
        self.andor.shutter(0, 1, 0, 0, 0)

        self.andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))
        self.recWidget.readyToRecord = True
        self.recWidget.recButton.setEnabled(True)

        # Initial image
        image = self.andor.most_recent_image16(self.shape)
        self.img.setImage(image, autoLevels=False, lut=self.lut)
        if update:
            self.hist.setLevels(np.min(image) - np.std(image),
                                np.max(image) + np.std(image))

            self.viewtimer.start(0)

        self.moleculeWidget.enableBox.setEnabled(True)

    def liveviewStop(self):
        self.viewtimer.stop()
        self.recWidget.readyToRecord = False

        # Turn off camera, close shutter
        idleMsg = 'Camera is idle, waiting for instructions.'
        if self.andor.status != idleMsg:
            self.andor.abort_acquisition()

        self.andor.shutter(0, 2, 0, 0, 0)
        self.img.setImage(np.zeros(self.shape), autoLevels=False)
        self.stabilizer.timer.start()

    def updateView(self):
        """ Image update while in Liveview mode
        """
        try:
            image = self.andor.most_recent_image16(self.shape)
            if self.moleculeWidget.enabled:
                self.moleculeWidget.graph.update(image)
            self.img.setImage(image, autoLevels=False)

            if self.crosshair.showed:
                ycoord = int(np.round(self.crosshair.hLine.pos()[1]))
                xcoord = int(np.round(self.crosshair.vLine.pos()[0]))
                self.xProfile.setData(image[:, ycoord])
                self.yProfile.setData(image[xcoord])

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
        self.fpsBox.setText('%0.2f fps' % self.fps)

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
        self.viewtimer.stop()
        self.stabilizer.timer.stop()
        self.stabilizerThread.terminate()

        # Turn off camera, close shutter
        if self.andor.status != 'Camera is idle, waiting for instructions.':
            self.andor.abort_acquisition()
        self.andor.shutter(0, 2, 0, 0, 0)

        self.laserWidgets.closeEvent(*args, **kwargs)
        self.focusWidget.closeEvent(*args, **kwargs)
        super().closeEvent(*args, **kwargs)


if __name__ == '__main__':

    app = QtGui.QApplication([])

    with instruments.Camera('andor.ccd.CCD') as andor, \
            instruments.Laser('mpb.vfl.VFL', 'COM11') as redlaser, \
            instruments.Laser('rgblasersystems.minilasevo.MiniLasEvo', 'COM7') as bluelaser, \
            instruments.Laser('laserquantum.ventus.Ventus', 'COM13') as greenlaser, \
            instruments.ScanZ(12) as scanZ:
            # instruments.DAQ() as DAQ, ScanZ(12) as scanZ:

        print(andor.idn)
        print(redlaser.idn)
        print(bluelaser.idn)
        print(greenlaser.idn)
#        print(DAQ.idn)
        print('Prior Z stage')

        win = TormentaGUI()
        win.show()

        app.exec_()
