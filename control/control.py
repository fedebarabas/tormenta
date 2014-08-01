# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

import numpy as np

from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import pyqtgraph.parametertree.parameterTypes as pTypes
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem
from pyqtgraph.parametertree import registerParameterType

import time
import h5py as hdf

from lantz.drivers.andor.ccd import CCD
from lantz import Q_

degC = Q_(1, 'degC')
us = Q_(1, 'us')
MHz = Q_(1, 'MHz')
s = Q_(1, 's')

lastTime = ptime.time()
fps = None

app = QtGui.QApplication([])


def SetCameraDefaults(camera):
    camera.readout_mode = 'Image'
    camera.trigger_mode = 'Internal'
    camera.amp_typ = 0
    camera.horiz_shift_speed = 0
    camera.vert_shift_speed = 0
#    camera.shutter(0, 5, 0, 0, 0)   # Uncomment when using for real
    camera.set_n_accum(1)                 # No accumulation of exposures
    camera.set_accum_time(0 * s)          # Minimum accumulation and kinetic
    camera.set_kinetic_cycle_time(0 * s)  # times


class TemperatureStabilizer(QtCore.QObject):

    def __init__(self, camera, *args, **kwargs):
        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.camera = camera

    def start(self):
        self.camera.temperature_setpoint = -30
        self.camera.cooler_on = True
        stable = 'Temperature has stabilized at set point.'
        print('Temperature set point =', self.camera.temperature_setpoint)
        while self.camera.temperature_status != stable:
            print("Current temperature:",
                  np.round(self.camera.temperature, 1))
            time.sleep(30)
        print('Temperature has stabilized at set point')


class TormentaGUI(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):
        global andor
        print(andor.idn)

        super(QtGui.QMainWindow, self).__init__(*args, **kwargs)

        # Camera configuration
        SetCameraDefaults(andor)
        self.ishape = list(andor.detector_shape)
        self.iorigin = [1, 1]
        andor.set_image(shape=self.ishape, p_0=self.iorigin)
        self.t_exp = 0.1 * s
        andor.set_exposure_time(self.t_exp)
        timings = andor.acquisition_timings
        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings

        self.setWindowTitle('Tormenta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Image Widget
        # TODO: redefine axis ticks
        imagewidget = pg.GraphicsLayoutWidget()
        self.p1 = imagewidget.addPlot()
        self.p1.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        self.p1.getViewBox().setLimits(xMin=-0.5, xMax=self.ishape[0] - 0.5,
                                       yMin=-0.5, yMax=self.ishape[1] - 0.5)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        self.p1.addItem(self.img)
        self.p1.setAspectLocked(True)
        self.p1.setRange(xRange=(-0.5, self.ishape[0] - 0.5),
                         yRange=(-0.5, self.ishape[1] - 0.5), padding=0)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.autoHistogramRange = False
        imagewidget.addItem(self.hist)
        self.fpsbox = QtGui.QLabel()

        # Liveview functionality
        LVButton = QtGui.QPushButton('LiveView')
        LVButton.pressed.connect(self.liveview)
        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateview)

        # Record functionality
        self.j = 0      # Image counter for recordings
        self.n = 100    # Number of expositions in recording
        self.filename = 'prueba.hdf5'

        # Temperature stabilization functionality
        StabButton = QtGui.QPushButton('Stabilize Temperature')
        self.Stabilizer = TemperatureStabilizer(andor)
        self.StabilizerThread = QtCore.QThread()
        self.Stabilizer.moveToThread(self.StabilizerThread)
        StabButton.clicked.connect(self.StabilizerThread.start)
        self.StabilizerThread.started.connect(self.Stabilizer.start)

        # Parameter tree for the camera configuration
        params = [{'name': 'Image frame', 'type': 'group', 'children': [
                   {'name': 'x_start', 'type': 'int',
                    'value': self.iorigin[0]},
                   {'name': 'y_start', 'type': 'int',
                    'value': self.iorigin[1]},
                   {'name': 'x_size', 'type': 'int',
                    'value': self.ishape[0]},
                   {'name': 'y_size', 'type': 'int',
                    'value': self.ishape[1]},
                   {'name': 'Update', 'type': 'action'},
                   ]},
                  {'name': 'Timings', 'type': 'group', 'children': [
                   {'name': 'Set exposure time', 'type': 'float',
                    'value': self.t_exp.magnitude,
                    'siPrefix': True, 'suffix': 's'},
                   {'name': 'Real exposure time', 'type': 'float',
                    'value': self.t_exp_real.magnitude, 'readonly': True,
                    'siPrefix': True, 'suffix': 's'},
                   {'name': 'Real accumulation time', 'type': 'float',
                    'value': self.t_acc_real.magnitude, 'readonly': True,
                    'siPrefix': True, 'suffix': 's'},
                   ]},
                  {'name': 'Recording', 'type': 'group', 'children': [
                   {'name': 'Number of expositions', 'type': 'int',
                    'value': self.n},
                   {'name': 'Filename', 'type': 'str',
                    'value': self.filename},
                   {'name': 'Start', 'type': 'action'},
                   ]}]

        self.p = Parameter.create(name='params', type='group', children=params)
        t = ParameterTree()
        t.setParameters(self.p, showTop=False)

        # Signals from the parameter tree
        FrameUpdateButton = self.p.param('Image frame').param('Update')
        FrameUpdateButton.sigStateChanged.connect(self.UpdateFrame)

        ExpPar = self.p.param('Timings').param('Set exposure time')
        ExpPar.sigValueChanged.connect(self.SetExposure)

        RecButton = self.p.param('Recording').param('Start')
        RecButton.sigStateChanged.connect(self.Record)

        # TODO: signal from the filename, folder

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.addWidget(StabButton, 5, 2)
        layout.addWidget(LVButton, 4, 3)
        layout.addWidget(imagewidget, 1, 2, 3, 3)
        layout.addWidget(self.fpsbox, 0, 4)
        layout.addWidget(t, 1, 1)

    def SetExposure(self):
        """ Method to change the exposure time setting
        """
        SetExpPar = self.p.param('Timings').param('Set exposure time')
        self.t_exp = SetExpPar.value() * s

        if andor.status != ('Camera is idle, '
                            'waiting for instructions.'):
            self.viewtimer.stop()
            andor.abort_acquisition()

        andor.set_exposure_time(self.t_exp)
        self.UpdateTimings()
        andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp.magnitude, 1)))
        self.viewtimer.start(0)

    def UpdateFrame(self):
        """ Method to change the image frame size and position in the sensor
        """
        ImgFramePar = self.p.param('Image frame')
        self.iorigin[0] = ImgFramePar.param('x_start').value()
        self.iorigin[1] = ImgFramePar.param('y_start').value()
        self.ishape[0] = ImgFramePar.param('x_size').value()
        self.ishape[1] = ImgFramePar.param('y_size').value()

        if andor.status != ('Camera is idle, '
                            'waiting for instructions.'):
            self.viewtimer.stop()
            andor.abort_acquisition()

        andor.set_image(shape=(self.ishape[0], self.ishape[1]),
                        p_0=(self.iorigin[0], self.iorigin[1]))
        self.p1.setRange(xRange=(-0.5, self.ishape[0] - 0.5),
                         yRange=(-0.5, self.ishape[1] - 0.5),
                         padding=0)
        self.p1.getViewBox().setLimits(xMin=-0.5, xMax=self.ishape[0] - 0.5,
                                       yMin=-0.5, yMax=self.ishape[1] - 0.5)
        self.UpdateTimings()
        andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp.magnitude, 1)))
        self.viewtimer.start(0)

    def UpdateTimings(self):
        """ Update the real exposition and accumulation times in the parameter
        tree.
        """
        timings = andor.acquisition_timings
        self.t_exp_real, self.t_acc_real, self.t_kin_real = timings
        RealExpPar = self.p.param('Timings').param('Real exposure time')
        RealAccPar = self.p.param('Timings').param('Real accumulation time')
        RealExpPar.setValue(self.t_exp_real.magnitude)
        RealAccPar.setValue(self.t_acc_real.magnitude)

    def liveview(self):
        """ Image live view when not recording
        """

        if andor.status != 'Camera is idle, waiting for instructions.':
            andor.abort_acquisition()

        andor.acquisition_mode = 'Run till abort'
        andor.free_int_mem()

        andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp.magnitude, 1)))
        idata = andor.most_recent_image16(self.ishape)

        # Initial image and histogram
        self.img.setImage(idata)
        self.hist.setLevels(0.8 * idata.min(), 1.2 * idata.max())
        self.viewtimer.start(0)

    # Image update while in liveview mode
    def updateview(self):
        global lastTime, fps
        image = andor.most_recent_image16(self.ishape)
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

    def Record(self):

        # Data storing
        self.store_file = hdf.File(self.filename, "w")
        self.store_file.create_dataset(name='data',
                                       shape=(self.n,
                                              self.ishape[0], self.ishape[1]),
                                       fillvalue=0.0)
        self.stack = self.store_file['data']

        # Acquisition preparation
        if andor.status != 'Camera is idle, waiting for instructions.':
            andor.abort_acquisition()

        andor.free_int_mem()
        andor.acquisition_mode = 'Kinetics'
        andor.set_n_kinetics(self.n)
        andor.start_acquisition()
        time.sleep(np.min((5 * self.t_exp.magnitude, 1)))

        # Stop the QTimer that updates the image with incoming data from the
        # 'Run till abort' acquisition mode.
        self.viewtimer.stop()
        QtCore.QTimer.singleShot(1, self.UpdateWhileRec)

    def UpdateWhileRec(self):
        global lastTime, fps

        if andor.n_images_acquired > self.j:
            i, self.j = andor.new_images_index
            self.stack[i - 1:self.j] = andor.images16(i, self.j, self.ishape,
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
            self.store_file.close()
            self.liveview()

    def closeEvent(self, *args, **kwargs):
        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)
        self.viewtimer.stop()
        andor

if __name__ == '__main__':

    from lantz import Q_
    s = Q_(1, 's')

    with CCD() as andor:

        win = TormentaGUI()
        win.show()

        app.exec_()
