# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

import numpy as np
import os
import time

from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.parametertree import Parameter, ParameterTree

import h5py as hdf
import tifffile as tiff     # http://www.lfd.uci.edu/~gohlke/pythonlibs/#vlfd
from lantz import Q_
from instruments import Laser, Camera, ScanZ, DAQ
from lasercontrol import LaserWidget
from focus import FocusWidget


# TODO: Implement cropped sensor mode in case we want higher framerates
# TODO: limits in histogram
# TODO: log en histograma para single molecule
class CamParamTree(ParameterTree):
    """ Making the ParameterTree for configuration of the camera during imaging
    """

    global andor

    def __init__(self, *args, **kwargs):
        super(CamParamTree, self).__init__(*args, **kwargs)

        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': andor.idn.split(',')[0]},
                  {'name': 'Image frame', 'type': 'group', 'children': [
                      {'name': 'Size', 'type': 'list',
                       'values': ['Full chip', '256x256', '128x128', '64x64',
                                  'Custom']},
                      {'name': 'Apply', 'type': 'action'}]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                      {'name': 'Frame Transfer Mode', 'type': 'bool',
                       'value': False},
                      {'name': 'Horizontal readout rate', 'type': 'list',
                       'values': andor.HRRates},
                      {'name': 'Vertical pixel shift', 'type': 'group',
                       'children': [
                           {'name': 'Speed', 'type': 'list',
                            'values': andor.vertSpeeds[::-1],
                            'value':andor.vertSpeeds[1]},
                           {'name': 'Clock voltage amplitude',
                            'type': 'list', 'values': andor.vertAmps}]},
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
                       'values': list(andor.PreAmps)},
                      {'name': 'EM gain', 'type': 'int', 'value': 1,
                       'limits': (0, andor.EM_gain_range[1])}]},
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
        self._editable = True

        self.p.param('Image frame').param('Size')

    @property
    def editable(self):
        return self._editable

    @editable.setter
    def editable(self, value):
        self._editable = value
        value = not(value)
        self.p.param('Image frame').param('Size').setReadonly(value)
        timeParams = self.p.param('Timings')
        timeParams.param('Frame Transfer Mode').setReadonly(value)
        timeParams.param('Horizontal readout rate').setReadonly(value)
        timeParams.param('Set exposure time').setReadonly(value)
        vpsParams = timeParams.param('Vertical pixel shift')
        vpsParams.param('Speed').setReadonly(True)
        vpsParams.param('Clock voltage amplitude').setReadonly(value)
        gainParams = self.p.param('Gain')
        gainParams.param('Pre-amp gain').setReadonly(value)
        gainParams.param('EM gain').setReadonly(value)


# TODO: get record methods as RecordingWidget methods
class RecordingWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):
        super(RecordingWidget, self).__init__(*args, **kwargs)

        recTitle = QtGui.QLabel('<h2><strong>Recording settings</strong></h2>')
        recTitle.setTextFormat(QtCore.Qt.RichText)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.currentFrame = QtGui.QLabel('0 /')
        self.numExpositionsEdit = QtGui.QLineEdit('100')
        self.folderEdit = QtGui.QLineEdit(os.getcwd())
        self.filenameEdit = QtGui.QLineEdit('filename')
        self.formatBox = QtGui.QComboBox()
        self.formatBox.addItems(['tiff', 'hdf5'])

        self.snapButton = QtGui.QPushButton('Snap')
        self.snapButton.setEnabled(False)
        self.snapButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.recButton = QtGui.QPushButton('REC')
        self.recButton.setCheckable(True)
        self.recButton.setEnabled(False)
        self.recButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                     QtGui.QSizePolicy.Expanding)

        recGrid = QtGui.QGridLayout()
        self.setLayout(recGrid)
        recGrid.addWidget(recTitle, 0, 0, 1, 3)
        recGrid.addWidget(QtGui.QLabel('Number of expositions'), 5, 0)
        recGrid.addWidget(self.currentFrame, 5, 1)
        recGrid.addWidget(self.numExpositionsEdit, 5, 2)
        recGrid.addWidget(QtGui.QLabel('Folder'), 1, 0, 1, 2)
        recGrid.addWidget(self.folderEdit, 2, 0, 1, 3)
        recGrid.addWidget(QtGui.QLabel('Filename'), 3, 0, 1, 2)
        recGrid.addWidget(self.filenameEdit, 4, 0, 1, 2)
        recGrid.addWidget(self.formatBox, 4, 2)
        recGrid.addWidget(self.snapButton, 1, 3, 2, 1)
        recGrid.addWidget(self.recButton, 3, 3, 3, 1)

        recGrid.setColumnMinimumWidth(0, 200)

        self._editable = True

    def nExpositions(self):
        return int(self.numExpositionsEdit.text())

    def folder(self):
        return self.folderEdit.text()

    def filename(self):
        return self.filenameEdit.text()

    @property
    def editable(self):
        return self._editable

    @editable.setter
    def editable(self, value):
        self.snapButton.setEnabled(value)
        self.folderEdit.setEnabled(value)
        self.filenameEdit.setEnabled(value)
        self.numExpositionsEdit.setEnabled(value)
        self.formatBox.setEnabled(value)
        self._editable = value


class TemperatureStabilizer(QtCore.QObject):

    def __init__(self, parameter, *args, **kwargs):

        global andor

        super(TemperatureStabilizer, self).__init__(*args, **kwargs)
        self.parameter = parameter
        self.setPointPar = self.parameter.param('Set point')
        self.setPointPar.sigValueChanged.connect(self.updateTemp)

    def updateTemp(self):
        andor.temperature_setpoint = Q_(self.setPointPar.value(), 'degC')

    def start(self):
        self.updateTemp()
        andor.cooler_on = True
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def update(self):
        stable = 'Temperature has stabilized at set point.'
        if andor.temperature_status != stable:
            CurrTempPar = self.parameter.param('Current temperature')
            CurrTempPar.setValue(np.round(andor.temperature.magnitude, 1))
            self.parameter.param('Status').setValue(andor.temperature_status)
            time.sleep(10)
        else:
            self.timer.stop()


# Check for same name conflict
def getUniqueName(name):

    n = 1
    while os.path.exists(name):
        if n > 1:
            name = name.replace('_{}.'.format(n - 1), '_{}.'.format(n))
        else:
            names = os.path.splitext(name)
            name = names[0] + '_{}'.format(n) + names[1]
        n += 1

    return name


class TormentaGUI(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        global andor

        super(TormentaGUI, self).__init__(*args, **kwargs)
        self.setWindowTitle('Tormenta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        self.s = Q_(1, 's')
        self.lastTime = ptime.time()
        self.fps = None

        self.tree = CamParamTree()

        # Frame signals
        self.shape = andor.detector_shape
        frameParam = self.tree.p.param('Image frame')
        frameParam.param('Size').sigValueChanged.connect(self.updateFrame)

        # Exposition signals
        changeExposure = lambda: self.changeParameter(self.setExposure)
        TimingsPar = self.tree.p.param('Timings')
        self.ExpPar = TimingsPar.param('Set exposure time')
        self.ExpPar.sigValueChanged.connect(changeExposure)
        self.FTMPar = TimingsPar.param('Frame Transfer Mode')
        self.FTMPar.sigValueChanged.connect(changeExposure)
        self.HRRatePar = TimingsPar.param('Horizontal readout rate')
        self.HRRatePar.sigValueChanged.connect(changeExposure)
        vertShiftPar = TimingsPar.param('Vertical pixel shift')
        self.vertShiftSpeedPar = vertShiftPar.param('Speed')
        self.vertShiftSpeedPar.sigValueChanged.connect(changeExposure)
        self.vertShiftAmpPar = vertShiftPar.param('Clock voltage amplitude')
        self.vertShiftAmpPar.sigValueChanged.connect(changeExposure)
        changeExposure()    # Set default values

        # Gain signals
        self.PreGainPar = self.tree.p.param('Gain').param('Pre-amp gain')
        updateGain = lambda: self.changeParameter(self.setGain)
        self.PreGainPar.sigValueChanged.connect(updateGain)
        self.GainPar = self.tree.p.param('Gain').param('EM gain')
        self.GainPar.sigValueChanged.connect(updateGain)
        updateGain()        # Set default values

        # Recording signals
        self.dataname = 'data'      # In case I need a QLineEdit for this
        self.recWidget = RecordingWidget()
        self.recWidget.recButton.clicked.connect(self.record)
        self.recWidget.snapButton.clicked.connect(self.snap)

        # Image Widget
        # TODO: redefine axis ticks
        self.shape = andor.detector_shape
        imagewidget = pg.GraphicsLayoutWidget()
        self.p1 = imagewidget.addPlot()
        self.p1.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.p1.addItem(self.img)
        self.p1.getViewBox().setAspectLocked(True)
        self.hist = pg.HistogramLUTItem()
        self.hist.gradient.loadPreset('yellowy')
        self.hist.setImageItem(self.img)
        self.hist.autoHistogramRange = False
        imagewidget.addItem(self.hist)

        # TODO: x, y profiles
        self.fpsBox = QtGui.QLabel()
        self.gridBox = QtGui.QCheckBox('Show grid')
        self.gridBox.stateChanged.connect(self.toggleGrid)

        # Initial camera configuration taken from the parameter tree
        andor.set_exposure_time(self.ExpPar.value() * self.s)
        self.adjustFrame()
        self.updateTimings()

        # Liveview functionality
        self.liveviewButton = QtGui.QPushButton('Liveview')
        self.liveviewButton.setCheckable(True)
        self.liveviewButton.clicked.connect(self.liveview)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateView)

        # Temperature stabilization functionality
        self.TempPar = self.tree.p.param('Temperature')
        self.stabilizer = TemperatureStabilizer(self.TempPar)
        self.stabilizerThread = QtCore.QThread()
        self.stabilizer.moveToThread(self.stabilizerThread)
        self.stabilizerThread.started.connect(self.stabilizer.start)
        self.stabilizerThread.start()

        # Laser control widget
        self.laserWidgets = LaserWidget((redlaser, bluelaser, greenlaser))
        self.focusWidget = FocusWidget(DAQ, scanZ)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.setColumnMinimumWidth(0, 400)
        layout.setColumnMinimumWidth(1, 800)
        layout.setColumnMinimumWidth(2, 200)
        layout.setRowMinimumHeight(0, 150)
        layout.setRowMinimumHeight(1, 320)
        layout.addWidget(self.tree, 0, 0, 2, 1)
        layout.addWidget(self.liveviewButton, 2, 0)
        layout.addWidget(self.recWidget, 3, 0, 2, 1)
        layout.addWidget(imagewidget, 0, 1, 4, 3)
        layout.addWidget(self.fpsBox, 4, 1)
        layout.addWidget(self.gridBox, 4, 2)
        layout.addWidget(self.laserWidgets, 0, 4)
        layout.addWidget(self.focusWidget, 1, 4)

    def changeParameter(self, function):
        """ This method is used to change those camera properties that need
        the camera to be idle to be able to be adjusted.
        """
        status = andor.status
        if status != ('Camera is idle, waiting for instructions.'):
            self.viewtimer.stop()
            andor.abort_acquisition()

        function()

        if status != ('Camera is idle, waiting for instructions.'):
            andor.start_acquisition()
            time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))
            self.viewtimer.start(0)

    def setGain(self):
        """ Method to change the pre-amp gain and main gain of the EMCCD
        """
        PreAmpGain = self.PreGainPar.value()
        n = np.where(andor.PreAmps == PreAmpGain)[0][0]
        andor.preamp = n
        andor.EM_gain = self.GainPar.value()

    def setExposure(self):
        """ Method to change the exposure time setting
        """
        andor.set_exposure_time(self.ExpPar.value() * self.s)
        andor.frame_transfer_mode = self.FTMPar.value()
        n_hrr = np.where(np.array([item.magnitude for item in andor.HRRates])
                         == self.HRRatePar.value().magnitude)[0][0]
        andor.horiz_shift_speed = n_hrr

        n_vss = np.where(np.array([item.magnitude
                                  for item in andor.vertSpeeds])
                         == self.vertShiftSpeedPar.value().magnitude)[0][0]
        andor.vert_shift_speed = n_vss

        n_vsa = np.where(np.array(andor.vertAmps) ==
                         self.vertShiftAmpPar.value())[0][0]
        andor.set_vert_clock(n_vsa)

        self.updateTimings()

    """ Grid methods """
    def showGrid(self):
        self.yline1 = pg.InfiniteLine(pos=0.25*self.shape[0], pen='y')
        self.yline2 = pg.InfiniteLine(pos=0.50*self.shape[0], pen='y')
        self.yline3 = pg.InfiniteLine(pos=0.75*self.shape[0], pen='y')
        self.xline1 = pg.InfiniteLine(pos=0.25*self.shape[1], pen='y', angle=0)
        self.xline2 = pg.InfiniteLine(pos=0.50*self.shape[1], pen='y', angle=0)
        self.xline3 = pg.InfiniteLine(pos=0.75*self.shape[1], pen='y', angle=0)
        self.p1.getViewBox().addItem(self.xline1)
        self.p1.getViewBox().addItem(self.xline2)
        self.p1.getViewBox().addItem(self.xline3)
        self.p1.getViewBox().addItem(self.yline1)
        self.p1.getViewBox().addItem(self.yline2)
        self.p1.getViewBox().addItem(self.yline3)

    def hideGrid(self):
        self.p1.getViewBox().removeItem(self.xline1)
        self.p1.getViewBox().removeItem(self.xline2)
        self.p1.getViewBox().removeItem(self.xline3)
        self.p1.getViewBox().removeItem(self.yline1)
        self.p1.getViewBox().removeItem(self.yline2)
        self.p1.getViewBox().removeItem(self.yline3)

    def toggleGrid(self, state):
        if state == QtCore.Qt.Checked:
            self.showGrid()
        else:
            self.hideGrid()

    def adjustFrame(self, shape=None, start=(1, 1)):
        """ Method to change the area of the CCD to be used and adjust the
        image widget accordingly.
        """
        if shape is None:
            shape = self.shape

        andor.set_image(shape=shape, p_0=start)
        self.p1.setRange(xRange=(-0.5, shape[0] - 0.5),
                         yRange=(-0.5, shape[1] - 0.5), padding=0)
        self.p1.getViewBox().setLimits(xMin=-0.5, xMax=shape[0] - 0.5,
                                       yMin=-0.5, yMax=shape[1] - 0.5,
                                       minXRange=4, minYRange=4)
        if self.gridBox.isChecked():
            self.hideGrid()
            self.showGrid()

        self.updateTimings()

    def updateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        frameParam = self.tree.p.param('Image frame')
        if frameParam.param('Size').value() == 'Custom':

            self.roi = pg.ROI((0.5 * self.shape[0] - 64,
                               0.5 * self.shape[1] - 64),
                              size=(128, 128), scaleSnap=True,
                              translateSnap=True, pen='y')
            self.roi.addScaleHandle((1, 0), (0, 1), lockAspect=True)
            self.p1.addItem(self.roi)

            # Signals
            applyParam = frameParam.param('Apply')
            applyParam.sigStateChanged.connect(self.customFrame)

        elif frameParam.param('Size').value() == 'Full chip':
            self.shape = andor.detector_shape
            self.changeParameter(self.adjustFrame)

        else:
            side = int(frameParam.param('Size').value().split('x')[0])
            start = (int(0.5 * (andor.detector_shape[0] - side)),
                     int(0.5 * (andor.detector_shape[1] - side)))
            self.shape = (side, side)
            self.changeParameter(lambda: self.adjustFrame(self.shape, start))

    def customFrame(self):

        self.shape = self.roi.size()
        start = self.roi.pos()

        self.changeParameter(lambda: self.adjustFrame(self.shape, start))
        self.roi.hide()

    def updateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree.
        """
        timings = andor.acquisition_timings
        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings
        timingsPar = self.tree.p.param('Timings')
        RealExpPar = timingsPar.param('Real exposure time')
        RealAccPar = timingsPar.param('Real accumulation time')
        EffFRPar = timingsPar.param('Effective frame rate')
        RealExpPar.setValue(self.t_exp_real.magnitude)
        RealAccPar.setValue(self.t_acc_real.magnitude)
        EffFRPar.setValue(1 / self.t_acc_real.magnitude)

    def liveview(self, update=True):
        """ Image live view when not recording
        """
        if self.liveviewButton.isChecked():
            if andor.status != 'Camera is idle, waiting for instructions.':
                andor.abort_acquisition()

            andor.acquisition_mode = 'Run till abort'
            andor.shutter(0, 1, 0, 0, 0)

            andor.start_acquisition()
            time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))
            self.recWidget.snapButton.setEnabled(True)
            self.recWidget.recButton.setEnabled(True)

            # Initial image
            image = andor.most_recent_image16(self.shape)
            self.img.setImage(image, autoLevels=False)
            if update:
                self.hist.setLevels(np.min(image) - np.std(image),
                                    np.max(image) + np.std(image))

            self.viewtimer.start(0)

        else:
            self.viewtimer.stop()

            # Turn off camera, close shutter
            if andor.status != 'Camera is idle, waiting for instructions.':
                andor.abort_acquisition()

            andor.shutter(0, 2, 0, 0, 0)
            self.img.setImage(np.zeros(self.shape), autoLevels=False)

    def updateView(self):
        """ Image update while in Liveview mode
        """
        try:
            image = andor.most_recent_image16(self.shape)
            self.img.setImage(image, autoLevels=False)
            now = ptime.time()
            dt = now - self.lastTime
            self.lastTime = now
            if self.fps is None:
                self.fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                self.fps = self.fps * (1-s) + (1.0/dt) * s
            self.fpsBox.setText('%0.2f fps' % self.fps)
        except:
            pass

    def snap(self):

        image = andor.most_recent_image16(self.shape)

        # Data storing
        self.folder = self.recWidget.folder()
        self.filename = self.recWidget.filename()
        self.format = self.recWidget.formatBox.currentText()
        self.savename = (os.path.join(self.folder, self.filename) + '.' +
                         self.format)

        if self.format == 'hdf5':
            self.store_file = hdf.File(getUniqueName(self.savename))
            self.store_file.create_dataset(name=self.dataname + '_snap',
                                           data=image)
            self.store_file.close()

        elif self.format == 'tiff':
            splitted = os.path.splitext(self.savename)
            snapname = splitted[0] + '_snap' + splitted[1]
            tiff.imsave(getUniqueName(snapname), image,
                        description=self.dataname, software='Tormenta')

    def record(self):

        if self.recWidget.recButton.isChecked():

            self.recWidget.editable = False
            self.tree.editable = False
            self.liveviewButton.setEnabled(False)

            # Frame counter
            self.j = 0

            # Data storing
            self.recPath = self.recWidget.folder()
            self.recFilename = self.recWidget.filename()
            self.n = self.recWidget.nExpositions()
            self.format = self.recWidget.formatBox.currentText()

            # Acquisition preparation
            if andor.status != 'Camera is idle, waiting for instructions.':
                andor.abort_acquisition()
            else:
                andor.shutter(0, 1, 0, 0, 0)

            andor.free_int_mem()
            andor.acquisition_mode = 'Kinetics'
            andor.set_n_kinetics(self.n)
            andor.start_acquisition()
            time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))

            # Stop the QTimer that updates the image with incoming data from
            # the 'Run till abort' acquisition mode.
            self.viewtimer.stop()

            self.savename = os.path.join(self.recPath,
                                         self.recFilename) + '.' + self.format

            if self.format == 'hdf5':
                """ Useful format for big data as it saves new frames in
                chunks. Therefore, you don't have the whole stack in memory."""

                self.store_file = hdf.File(getUniqueName(self.savename), "w")
                self.store_file.create_dataset(name=self.dataname,
                                               shape=(self.n,
                                                      self.shape[0],
                                                      self.shape[1]),
                                               fillvalue=0, dtype=np.uint16)
                self.stack = self.store_file[self.dataname]

            elif self.format == 'tiff':
                """ This format has the problem of placing the whole stack in
                memory before saving."""

                # TODO: Work with memmap
                self.stack = np.empty((self.n, self.shape[0], self.shape[1]),
                                      dtype=np.uint16)

            QtCore.QTimer.singleShot(1, self.updateWhileRec)

    def updateWhileRec(self):
        global lastTime, fps

        time.sleep(self.t_exp_real.magnitude)

        if andor.n_images_acquired > self.j:
            i, self.j = andor.new_images_index
            self.stack[i - 1:self.j] = andor.images16(i, self.j, self.shape,
                                                      1, self.n)
            self.img.setImage(self.stack[self.j - 1], autoLevels=False)
            self.recWidget.currentFrame.setText(str(self.j) + ' /')

            now = ptime.time()
            dt = now - self.lastTime
            self.lastTime = now
            if self.fps is None:
                self.fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                self.fps = self.fps * (1-s) + (1.0/dt) * s
            self.fpsBox.setText('%0.2f fps' % self.fps)

        if self.j < self.n and self.recWidget.recButton.isChecked():
            QtCore.QTimer.singleShot(0, self.updateWhileRec)

        else:
            self.endRecording()

    def endRecording(self):

        if self.format == 'hdf5':
            # TODO: Crop results to self.j frames

            # Saving parameters as data attributes in the HDF5 file
            dset = self.store_file[self.dataname]
            dset.attrs['Date'] = time.strftime("%Y-%m-%d")
            dset.attrs['Time'] = time.strftime("%H:%M:%S")
            attrs = []
            for ParName in self.tree.p.getValues():
                Par = self.tree.p.param(str(ParName))
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
                                    attrs.append((str(ssParName),
                                                  ssPar.value()))

            for item in attrs:
                dset.attrs[item[0]] = item[1]

            self.store_file.close()

        elif self.format == 'tiff':

            tiff.imsave(getUniqueName(self.savename), self.stack[0:self.j],
                        description=self.dataname, software='Tormenta')

        self.j = 0                                  # Reset counter
        self.recWidget.recButton.setChecked(False)
        self.recWidget.editable = True
        self.tree.editable = True
        self.liveviewButton.setEnabled(True)
        self.liveview(update=False)

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
        self.viewtimer.stop()
        self.stabilizer.timer.stop()
        self.stabilizerThread.terminate()

        # Turn off camera, close shutter
        if andor.status != 'Camera is idle, waiting for instructions.':
            andor.abort_acquisition()
        andor.shutter(0, 2, 0, 0, 0)

        self.laserWidgets.closeEvent(*args, **kwargs)
        self.focusWidget.closeEvent(*args, **kwargs)
        super(TormentaGUI, self).closeEvent(*args, **kwargs)


if __name__ == '__main__':

    app = QtGui.QApplication([])

    with Camera('andor.ccd.CCD') as andor, \
            Laser('mpb.vfl.VFL', 'COM11') as redlaser, \
            Laser('cobolt.cobolt0601.Cobolt0601', 'COM4') as bluelaser, \
            Laser('laserquantum.ventus.Ventus', 'COM10') as greenlaser, \
            DAQ() as DAQ, ScanZ(12) as scanZ:

        print(andor.idn)
        print(redlaser.idn)
        print(bluelaser.idn)
        print(greenlaser.idn)
        print(DAQ.idn)
        print('Prior Z stage')

        win = TormentaGUI()
        win.show()

        app.exec_()
