# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

import numpy as np
import os

from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.parametertree import Parameter, ParameterTree

import time
import h5py as hdf

# Lantz drivers
from lantz.drivers.andor.ccd import CCD
#from lantz.drivers.cobolt import Cobolt0601
from lantz.drivers.mpb import VFL
from lantz import Q_

degC = Q_(1, 'degC')
us = Q_(1, 'us')
MHz = Q_(1, 'MHz')
s = Q_(1, 's')
mW = Q_(1, 'mW')

lastTime = ptime.time()
fps = None

app = QtGui.QApplication([])

# TODO: Implement cropped sensor mode in case we want higher framerates


class RecordingWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):
        super(QtGui.QFrame, self).__init__(*args, **kwargs)

        recTitle = QtGui.QLabel('<h2><strong>Recording settings</strong></h2>')
        recTitle.setTextFormat(QtCore.Qt.RichText)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        recGrid = QtGui.QGridLayout()
        self.setLayout(recGrid)
        numExpositions = QtGui.QLabel('Number of expositions')
        self.numExpositionsEdit = QtGui.QLineEdit('100')
        folderLabel = QtGui.QLabel('Folder')
        self.folderEdit = QtGui.QLineEdit(os.getcwd())
        filenameLabel = QtGui.QLabel('Filename')
        self.filenameEdit = QtGui.QLineEdit('filename.hdf5')
        self.snapButton = QtGui.QPushButton('Snap')
        self.snapButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.recButton = QtGui.QPushButton('REC')
        self.recButton.setCheckable(True)
        self.recButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                     QtGui.QSizePolicy.Expanding)
        self.convertButton = QtGui.QPushButton('Convert to .tiff')

        recGrid.addWidget(recTitle, 0, 0, 1, 3)
        recGrid.addWidget(numExpositions, 1, 0)
        recGrid.addWidget(self.numExpositionsEdit, 1, 1)
        recGrid.addWidget(folderLabel, 2, 0, 1, 2)
        recGrid.addWidget(self.folderEdit, 3, 0, 1, 2)
        recGrid.addWidget(filenameLabel, 4, 0, 1, 2)
        recGrid.addWidget(self.filenameEdit, 5, 0, 1, 2)
        recGrid.addWidget(self.snapButton, 1, 2, 2, 1)
        recGrid.addWidget(self.recButton, 3, 2, 2, 1)
        recGrid.addWidget(self.convertButton, 5, 2)

    def nExpositions(self):
        return int(self.numExpositionsEdit.text())

    def folder(self):
        return self.folderEdit.text()

    def filename(self):
        return self.filenameEdit.text()


class UpdatePowers(QtCore.QObject):

    def __init__(self, laserwidget, *args, **kwargs):

        global redlaser  # , bluelaser

        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.widget = laserwidget

    def update(self):
        redpower = str(np.round(redlaser.power))
        self.widget.RedControl.PowerIndicator.setText(redpower)
        time.sleep(1)
        QtCore.QTimer.singleShot(1, self.update)


class LaserWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):

        global redlaser  # , bluelaser

        super(QtGui.QFrame, self).__init__(*args, **kwargs)

        LaserTitle = QtGui.QLabel('<h2>Laser control</h2>')
        LaserTitle.setTextFormat(QtCore.Qt.RichText)

        self.RedControl = LaserControl(redlaser,
                                       '<h3>MPB 642nm 1500mW</h3>',
                                       color=(255, 11, 0), prange=(150, 1500),
                                       tickInterval=100, singleStep=10)

#        self.BlueControl = LaserControl(bluelaser,
#                                        '<h3>Cobolt 405nm 100mW</h3>',
#                                        color=(73, 0, 188),
#                                        prange=(0, 100),
#                                        tickInterval=10, singleStep=1)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(LaserTitle, 0, 0)
        grid.addWidget(self.RedControl, 1, 1)
#        grid.addWidget(self.BlueControl, 1, 0)

        # Current power update routine
        self.updatePowers = UpdatePowers(self)
        self.updateThread = QtCore.QThread()
        self.updatePowers.moveToThread(self.updateThread)
        self.updateThread.start()
        self.updateThread.started.connect(self.updatePowers.update)


class LaserControl(QtGui.QWidget):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 *args, **kwargs):
        super(QtGui.QWidget, self).__init__(*args, **kwargs)
        self.laser = laser
        self.laser.power_sp = prange[0] * mW

        self.setGeometry(10, 10, 30, 150)

        self.Name = QtGui.QLabel()
        self.Name.setTextFormat(QtCore.Qt.RichText)
        self.Name.setText(name)

        self.PowerIndicator = QtGui.QLineEdit('0')
        self.PowerIndicator.setReadOnly(True)

        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))

        self.EnableButton = QtGui.QPushButton('ON')
        style = "background-color: rgb{}".format(color)
        self.EnableButton.setStyleSheet(style)
        self.EnableButton.setCheckable(True)

        self.Slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.Slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.Slider.setMinimum(prange[0])
        self.Slider.setMaximum(prange[1])
        self.Slider.setTickInterval(tickInterval)
        self.Slider.setSingleStep(singleStep)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.Name, 0, 0, 1, 2)
        grid.addWidget(self.PowerIndicator, 2, 0)
        grid.addWidget(self.setPointEdit, 3, 0)
        grid.addWidget(self.EnableButton, 4, 0)
        grid.addWidget(self.Slider, 1, 1, 6, 1)

        grid.setRowMinimumHeight(1, 50)
        grid.setRowMinimumHeight(6, 50)

        # Connections
        self.EnableButton.toggled.connect(self.toggleLaser)
        self.Slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

    def toggleLaser(self):
        if self.EnableButton.isChecked():
            self.laser.enabled = True
        else:
            self.laser.enabled = False

    def enableLaser(self):
        self.laser.enabled = True
        self.laser.power_sp = float(self.setPointEdit.text()) * mW

    def changeSlider(self, value):
        self.laser.power_sp = self.Slider.value() * mW
        self.setPointEdit.setText(str(self.laser.power_sp.magnitude))

    def changeEdit(self):
        self.laser.power_sp = float(self.setPointEdit.text()) * mW
        self.Slider.setValue(self.laser.power_sp.magnitude)


def SetCameraDefaults(camera):
    """ Initial camera's configuration
    """
    camera.readout_mode = 'Image'
    camera.trigger_mode = 'Internal'
    camera.preamp = 0
    camera.EM_advanced_enabled = False
    camera.EM_gain_mode = 'RealGain'
    camera.amp_typ = 0
    camera.vert_shift_speed = 0
    camera.set_n_accum(1)                 # No accumulation of exposures
    camera.set_accum_time(0 * s)          # Minimum accumulation and kinetic
    camera.set_kinetic_cycle_time(0 * s)  # times
    camera.horiz_shift_speed = 3


class TemperatureStabilizer(QtCore.QObject):

    def __init__(self, parameter, *args, **kwargs):

        global andor

        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.parameter = parameter

    def start(self):
        SetPointPar = self.parameter.param('Set point')
        andor.temperature_setpoint = SetPointPar.value() * degC
        andor.cooler_on = True
        stable = 'Temperature has stabilized at set point.'
        CurrTempPar = self.parameter.param('Current temperature')
        while andor.temperature_status != stable:
            CurrTempPar.setValue(np.round(andor.temperature.magnitude, 1))
            self.parameter.param('Status').setValue(andor.temperature_status)
            time.sleep(10)


class TormentaGUI(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        global andor  # , bluelaser

#        bluelaser.enabled = True

        super(QtGui.QMainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle('Tormenta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Lists needed for the parameter tree
        self.PreAmps = np.around([andor.true_preamp(n)
                                  for n in np.arange(andor.n_preamps)],
                                 decimals=1)
        self.HRRates = [andor.true_horiz_shift_speed(n)
                        for n in np.arange(andor.n_horiz_shift_speeds())]

        # Parameter tree for the camera configuration
        params = [{'name': 'Camera', 'type': 'str',
                   'value': andor.idn.split(',')[0]},
                  {'name': 'Image frame', 'type': 'group', 'children': [
                   {'name': 'x_start', 'type': 'int', 'suffix': 'px',
                    'value': 1},
                   {'name': 'y_start', 'type': 'int', 'suffix': 'px',
                    'value': 1},
                   {'name': 'x_size', 'type': 'int', 'suffix': 'px',
                    'value': andor.detector_shape[0]},
                   {'name': 'y_size', 'type': 'int', 'suffix': 'px',
                    'value': andor.detector_shape[1]},
                   {'name': 'Update', 'type': 'action'},
                   ]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                   {'name': 'Frame Transfer Mode', 'type': 'bool',
                    'value': False},
                   {'name': 'Horizontal readout rate', 'type': 'list',
                    'values': self.HRRates[::-1]},
                   {'name': 'Set exposure time', 'type': 'float',
                    'value': 0.1, 'limits': (0, andor.max_exposure.magnitude),
                    'siPrefix': True, 'suffix': 's'},
                   {'name': 'Real exposure time', 'type': 'float',
                    'value': 0, 'readonly': True, 'siPrefix': True,
                    'suffix': 's'},
                   {'name': 'Real accumulation time', 'type': 'float',
                    'value': 0, 'readonly': True, 'siPrefix': True,
                    'suffix': 's'},
                   {'name': 'Effective frame rate', 'type': 'float',
                    'value': 0, 'readonly': True, 'siPrefix': True,
                    'suffix': 'Hz'},
                   ]},
                  {'name': 'Gain', 'type': 'group', 'children': [
                   {'name': 'Pre-amp gain', 'type': 'list',
                    'values': list(self.PreAmps)},
                   {'name': 'EM gain', 'type': 'int', 'value': 1,
                    'limits': (0, andor.EM_gain_range[1])}
                   ]},
                  {'name': 'Temperature', 'type': 'group', 'children': [
                   {'name': 'Set point', 'type': 'int', 'value': -40,
                    'suffix': 'º', 'limits': (-80, 0)},
                   {'name': 'Current temperature', 'type': 'int',
                    'value': andor.temperature.magnitude, 'suffix': 'ºC',
                    'readonly': True},
                   {'name': 'Status', 'type': 'str', 'readonly': True,
                    'value': andor.temperature_status},
                   {'name': 'Stabilize', 'type': 'action'},
                   ]}]

        self.p = Parameter.create(name='params', type='group', children=params)
        tree = ParameterTree()
        tree.setParameters(self.p, showTop=False)

        # Frame signals
        FrameUpdateButton = self.p.param('Image frame').param('Update')
        UpdateFrame = lambda: self.ChangeParameter(self.UpdateFrame)
        FrameUpdateButton.sigStateChanged.connect(UpdateFrame)

        # Exposition signals
        self.TimingsPar = self.p.param('Timings')
        self.ExpPar = self.TimingsPar.param('Set exposure time')
        self.FTMPar = self.TimingsPar.param('Frame Transfer Mode')
        self.HRRatePar = self.TimingsPar.param('Horizontal readout rate')
        UpdateExposure = lambda: self.ChangeParameter(self.SetExposure)
        self.ExpPar.sigValueChanged.connect(UpdateExposure)
        self.FTMPar.sigValueChanged.connect(UpdateExposure)
        self.HRRatePar.sigValueChanged.connect(UpdateExposure)

        # Gain signals
        self.PreGainPar = self.p.param('Gain').param('Pre-amp gain')
        UpdateGain = lambda: self.ChangeParameter(self.SetGain)
        self.PreGainPar.sigValueChanged.connect(UpdateGain)
        self.GainPar = self.p.param('Gain').param('EM gain')
        self.GainPar.sigValueChanged.connect(UpdateGain)

        # Recording signals
        self.dataname = 'data'      # In case I need a QLineEdit for this
        self.recWidget = RecordingWidget()
        self.recWidget.recButton.clicked.connect(self.Record)
        self.recWidget.convertButton.clicked.connect(self.ConvertToRaw)

        # Image Widget
        # TODO: redefine axis ticks
        imagewidget = pg.GraphicsLayoutWidget()
        self.p1 = imagewidget.addPlot()
        self.p1.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.p1.addItem(self.img)
        self.p1.getViewBox().setAspectLocked(True)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.autoHistogramRange = False
        imagewidget.addItem(self.hist)

        self.fpsbox = QtGui.QLabel()

        # Initial camera configuration taken from the parameter tree
        SetCameraDefaults(andor)
        andor.set_exposure_time(self.ExpPar.value() * s)
        self.AdjustFrame()
        self.UpdateTimings()

        # Liveview functionality
        LVButton = QtGui.QPushButton('Liveview')
        LVButton.pressed.connect(self.Liveview)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.UpdateView)

        # Temperature stabilization functionality
        self.TempPar = self.p.param('Temperature')
        StabButton = self.TempPar.param('Stabilize')
        self.Stabilizer = TemperatureStabilizer(self.TempPar)
        self.StabilizerThread = QtCore.QThread()
        self.Stabilizer.moveToThread(self.StabilizerThread)
        StabButton.sigStateChanged.connect(self.StabilizerThread.start)
        self.StabilizerThread.started.connect(self.Stabilizer.start)

        self.LaserWidgets = LaserWidget()

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.setColumnMinimumWidth(1, 400)
        layout.setColumnMinimumWidth(2, 800)
        layout.setColumnMinimumWidth(3, 200)
        layout.setRowMinimumHeight(1, 150)
        layout.setRowMinimumHeight(2, 250)
        layout.addWidget(tree, 1, 1, 2, 1)
        layout.addWidget(LVButton, 3, 1)
        layout.addWidget(self.recWidget, 4, 1)
        layout.addWidget(imagewidget, 1, 2, 4, 1)
        layout.addWidget(self.fpsbox, 0, 2)
        layout.addWidget(self.LaserWidgets, 1, 3)

    def ChangeParameter(self, function):
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

    def SetGain(self):
        """ Method to change the pre-amp gain and main gain of the EMCCD
        """
        PreAmpGain = self.PreGainPar.value()
        n = np.where(self.PreAmps == PreAmpGain)[0][0]
        andor.preamp = n
        andor.EM_gain = self.GainPar.value()

    def SetExposure(self):
        """ Method to change the exposure time setting
        """
        andor.set_exposure_time(self.ExpPar.value() * s)
        andor.frame_transfer_mode = self.FTMPar.value()
        HRRate = self.HRRatePar.value()
        HRRatesMagnitude = np.array([item.magnitude for item in self.HRRates])
        n = np.where(HRRatesMagnitude == HRRate.magnitude)[0][0]
        andor.horiz_shift_speed = n
        self.UpdateTimings()

    def AdjustFrame(self):
        """ Method to change the area of the CCD to be used and adjust the
        image widget accordingly.
        """
        self.shape = [self.p.param('Image frame').param('x_size').value(),
                      self.p.param('Image frame').param('y_size').value()]
        self.p_0 = [self.p.param('Image frame').param('x_start').value(),
                    self.p.param('Image frame').param('y_start').value()]
        andor.set_image(shape=self.shape, p_0=self.p_0)
        self.p1.setRange(xRange=(-0.5, self.shape[0] - 0.5),
                         yRange=(-0.5, self.shape[1] - 0.5), padding=0)
        self.p1.getViewBox().setLimits(xMin=-0.5, xMax=self.shape[0] - 0.5,
                                       yMin=-0.5, yMax=self.shape[1] - 0.5,
                                       minXRange=4, minYRange=4)

    def UpdateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        self.AdjustFrame()
        self.UpdateTimings()

    def UpdateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree.
        """
        timings = andor.acquisition_timings
        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings
        RealExpPar = self.p.param('Timings').param('Real exposure time')
        RealAccPar = self.p.param('Timings').param('Real accumulation time')
        EffFRPar = self.p.param('Timings').param('Effective frame rate')
        RealExpPar.setValue(self.t_exp_real.magnitude)
        RealAccPar.setValue(self.t_acc_real.magnitude)
        EffFRPar.setValue(1 / self.t_acc_real.magnitude)

    def Liveview(self):
        """ Image live view when not recording
        """

        if andor.status != 'Camera is idle, waiting for instructions.':
            andor.abort_acquisition()

        andor.acquisition_mode = 'Run till abort'
        andor.free_int_mem()
        andor.shutter(0, 1, 0, 0, 0)

        andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp_real.magnitude, 1)))
        idata = andor.most_recent_image16(self.shape)

        # Initial image and histogram
        self.img.setImage(idata)
        self.hist.setLevels(0.8 * idata.min(), 1.2 * idata.max())
        self.viewtimer.start(0)

    def UpdateView(self):
        """ Image update while in Liveview mode
        """
        global lastTime, fps
        try:
            image = andor.most_recent_image16(self.shape)
            self.img.setImage(image, autoLevels=False)
            now = ptime.time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                fps = fps * (1-s) + (1.0/dt) * s
            self.fpsbox.setText('%0.2f fps' % fps)
        except:
            pass

    def Record(self):

        self.j = 0

        # Data storing
        self.folder = self.recWidget.folder()
        self.filename = self.recWidget.filename()
        self.n = self.recWidget.nExpositions()
        self.store_file = hdf.File(os.path.join(self.folder, self.filename),
                                   "w")
        self.store_file.create_dataset(name=self.dataname,
                                       shape=(self.n,
                                              self.shape[0],
                                              self.shape[1]),
                                       fillvalue=0.0)
        self.stack = self.store_file['data']

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

        # Stop the QTimer that updates the image with incoming data from the
        # 'Run till abort' acquisition mode.
        self.viewtimer.stop()
        QtCore.QTimer.singleShot(1, self.UpdateWhileRec)

    def UpdateWhileRec(self):
        global lastTime, fps

        if andor.n_images_acquired > self.j:
            i, self.j = andor.new_images_index
            self.stack[i - 1:self.j] = andor.images16(i, self.j, self.shape,
                                                      1, self.n)
            self.img.setImage(self.stack[self.j - 1], autoLevels=False)

            now = ptime.time()
            dt = now - lastTime
            lastTime = now
            if fps is None:
                fps = 1.0/dt
            else:
                s = np.clip(dt*3., 0, 1)
                fps = fps * (1-s) + (1.0/dt) * s
            self.fpsbox.setText('%0.2f fps' % fps)

        if self.j < self.n:
            QtCore.QTimer.singleShot(0, self.UpdateWhileRec)
        else:
            self.j = 0

            # Saving parameters as data attributes in the HDF5 file
            dset = self.store_file[self.dataname]
            dset.attrs['Date'] = time.strftime("%Y-%m-%d")
            dset.attrs['Time'] = time.strftime("%H:%M:%S")
            for ParamName in self.p.getValues():
                Param = self.p.param(str(ParamName))
                if not(Param.hasChildren()):
                    dset.attrs[str(ParamName)] = Param.value()
                for subParamName in Param.getValues():
                    subParam = Param.param(str(subParamName))
                    if subParam.type() != 'action':
                        dset.attrs[str(subParamName)] = subParam.value()

            self.store_file.close()
            self.recWidget.recButton.setChecked(False)
            self.Liveview()

    def ConvertToRaw(self):

        # TODO: implement this
        self.store_file = hdf.File(os.path.join(self.folder, self.filename),
                                   "r")

    def closeEvent(self, *args, **kwargs):

        # Stop running threads
        self.viewtimer.stop()
        self.LaserWidgets.updateThread.terminate()
        self.StabilizerThread.terminate()

        # Turn off camera, close shutter
        if andor.status != 'Camera is idle, waiting for instructions.':
            andor.abort_acquisition()
        andor.shutter(0, 2, 0, 0, 0)

#        laserOff(bluelaser, 10 * mW)
#        laserOff(redlaser, 150 * mW)
        redlaser.enabled = False
#        bluelaser.enabled = False

        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)


def laserOff(laser, mini):
    """ Lasers' shutting down protocol
    """
    if laser.power_sp > 2 * mini:
        while laser.power_sp > 2 * mini:
            ipower = laser.power_sp
            laser.power_sp = ipower - mini
            time.sleep(3)

        laser.power_sp = mini
        time.sleep(3)

    laser.enabled = False

if __name__ == '__main__':

    from lantz import Q_
    s = Q_(1, 's')

    with CCD() as andor, VFL('COM5') as redlaser:

        print(andor.idn)
        print(redlaser.idn)
#        print(bluelaser.idn)

        win = TormentaGUI()
        win.show()

        app.exec_()
