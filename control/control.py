# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import time

from lantz.drivers.andor.ccd import CCD

lastTime = ptime.time()
fps = None

app = QtGui.QApplication([])


def SetCameraDefaults(camera):
    camera.readout_mode = 'Image'
    camera.trigger_mode = 'Internal'
    camera.amp_typ = 0
    camera.horiz_shift_speed = 0
    camera.vert_shift_speed = 0
#    self.camera.shutter(0, 5, 0, 0, 0)   # Uncomment when using for real


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

        super(QtGui.QMainWindow, self).__init__(*args, **kwargs)

        self.camera = andor
        print(self.camera.idn)

        # Camera configuration
        SetCameraDefaults(self.camera)
        self.t_exp = 0.02 * s       # Exposition time
        self.ishape = self.camera.detector_shape
        self.iorigin = (1, 1)
        self.camera.set_exposure_time(self.t_exp)
        self.camera.set_image(shape=self.ishape, p_0=self.iorigin)

        self.setWindowTitle('Tormenta')
        self.cwidget = QtGui.QWidget()
        self.setCentralWidget(self.cwidget)

        # Image Widget
        # TODO: redefine axis ticks
        imagewidget = pg.GraphicsLayoutWidget()
        p1 = imagewidget.addPlot()
        p1.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        p1.getViewBox().setLimits(xMin=-0.5, xMax=self.ishape[0] - 0.5,
                                  yMin=-0.5, yMax=self.ishape[1] - 0.5)
        self.img = pg.ImageItem()
        self.img.translate(-0.5, -0.5)
        p1.addItem(self.img)
        p1.setAspectLocked(True)
        p1.setRange(xRange=(-0.5, self.ishape[0] - 0.5),
                    yRange=(-0.5, self.ishape[1] - 0.5), padding=0)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.hist.autoHistogramRange = False
        imagewidget.addItem(self.hist)

        # Widgets
        rec = QtGui.QPushButton('REC')
        StabButton = QtGui.QPushButton('Stabilize Temperature')
        LVButton = QtGui.QPushButton('LiveView')
        self.fpsbox = QtGui.QLabel()

        # Widgets' layout
        layout = QtGui.QGridLayout()
        self.cwidget.setLayout(layout)
        layout.addWidget(rec, 1, 0)
        layout.addWidget(StabButton, 2, 0)
        layout.addWidget(LVButton, 3, 0)
        layout.addWidget(imagewidget, 1, 2, 3, 1)
        layout.addWidget(self.fpsbox, 0, 2)

        self.viewtimer = QtCore.QTimer()
        self.viewtimer.timeout.connect(self.updateview)

        self.j = 0      # Image counter for recordings
        self.n = 100    # Number of expositions in recording
        rec.pressed.connect(self.record)

        LVButton.pressed.connect(self.liveview)

        self.Stabilizer = TemperatureStabilizer(self.camera)
        self.StabilizerThread = QtCore.QThread()
        self.Stabilizer.moveToThread(self.StabilizerThread)
        StabButton.clicked.connect(self.StabilizerThread.start)
        self.StabilizerThread.started.connect(self.Stabilizer.start)


    def show(self, *args, **kwargs):
        super(QtGui.QMainWindow, self).show(*args, **kwargs)

#        # Temperature stabilization
#        self.camera.temperature_setpoint = -30
#        self.camera.cooler_on = True
#        stable = 'Temperature has stabilized at set point.'
#        print('Temperature set point =', self.camera.temperature_setpoint)
#        while self.camera.temperature_status != stable:
#            print("Current temperature:",
#                  np.round(self.camera.temperature, 1))
#            time.sleep(30)
#        print('Temperature has stabilized at set point')

#        self.liveview()

    def closeEvent(self, *args, **kwargs):
        super(QtGui.QMainWindow, self).closeEvent(*args, **kwargs)
        self.viewtimer.stop()
        self.camera

    def liveview(self):
        """ Image live view when not recording
        """

        if self.camera.status != 'Camera is idle, waiting for instructions.':
            self.camera.abort_acquisition()

        self.camera.acquisition_mode = 'Run till abort'
        self.camera.free_int_mem()

        self.camera.start_acquisition()
        time.sleep(5 * self.t_exp.magnitude)
        idata = self.camera.most_recent_image16(self.camera.detector_shape)

        # Initial image and histogram
        self.img.setImage(idata)
        self.hist.setLevels(0.8 * idata.min(), 1.2 * idata.max())

        self.viewtimer.start(0)

    # Image update while in liveview mode
    def updateview(self):
        global lastTime, fps
        image = self.camera.most_recent_image16(self.camera.detector_shape)
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

    def record(self):
        self.stack = np.zeros((self.n, self.ishape[0], self.ishape[1]))

        # Acquisition preparation
        if self.camera.status != 'Camera is idle, waiting for instructions.':
            self.camera.abort_acquisition()

        self.camera.free_int_mem()
        self.camera.acquisition_mode = 'Kinetics'
        self.camera.set_n_kinetics(self.n)
        self.camera.start_acquisition()
        time.sleep(5 * self.t_exp.magnitude)

        # Stop the QTimer that updates the image with incoming data from the
        # 'Run till abort' acquisition mode.
        self.viewtimer.stop()
        QtCore.QTimer.singleShot(1, self.UpdateWhileRec)

    def UpdateWhileRec(self):
        global lastTime, fps

        if self.camera.n_images_acquired > self.j:
            i, self.j = self.camera.new_images_index
            self.stack[i - 1:self.j] = self.camera.images16(i, self.j,
                                                            self.ishape, 1,
                                                            self.n)
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
            self.liveview()

if __name__ == '__main__':

    from lantz import Q_
    s = Q_(1, 's')

    with CCD() as andor:

        win = TormentaGUI()
        win.show()

        app.exec_()
