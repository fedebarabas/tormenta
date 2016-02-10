# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:51:21 2014

@author: Federico Barabas
"""

import os

import numpy as np
import time
from PyQt4 import QtGui, QtCore
from lantz import Q_
from tormenta.control.instruments import daqStream


class UpdatePowers(QtCore.QObject):

    def __init__(self, laserwidget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = laserwidget

    def update(self):
        redPower = np.round(self.widget.redlaser.power.magnitude, 1)
        bluePower = np.round(self.widget.bluelaser.power.magnitude, 1)
        greenPower = np.round(self.widget.greenlaser.power.magnitude, 1)
        self.widget.redControl.powerIndicator.setText(str(redPower))
        self.widget.blueControl.powerIndicator.setText(str(bluePower))
        self.widget.greenControl.powerIndicator.setText(str(greenPower))
        time.sleep(1)
        QtCore.QTimer.singleShot(1, self.update)


class LaserWidget(QtGui.QFrame):

    def __init__(self, main, lasers, daq, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = main
        self.redlaser, self.bluelaser, self.greenlaser = lasers
        self.mW = Q_(1, 'mW')
        self.V = Q_(1, 'V')
        self.daq = daq
        self.daq.digital_IO[3] = True

        calibrationFilename = r'tormenta/control/calibration.txt'
        self.calibrationPath = os.path.join(os.getcwd(), calibrationFilename)

        self.redControl = LaserControl(self.redlaser, '<h3>642nm</h3>',
                                       color=(255, 11, 0), prange=(150, 1500),
                                       tickInterval=100, singleStep=10,
                                       daq=self.daq, port=1)

        self.blueControl = LaserControl(self.bluelaser, '<h3>405nm</h3>',
                                        color=(73, 0, 188), prange=(0, 53),
                                        tickInterval=5, singleStep=0.1)

        self.greenControl = LaserControl(self.greenlaser, '<h3>532nm</h3>',
                                         color=(80, 255, 0), prange=(0, 1500),
                                         tickInterval=10, singleStep=1,
                                         daq=self.daq, port=0, invert=False)

        self.shuttLasers = np.array([self.redControl, self.greenControl])
        self.controls = (self.blueControl, self.redControl, self.greenControl)

        self.findTirfButton = QtGui.QPushButton('Find TIRF (no anda)')
        self.setEpiButton = QtGui.QPushButton('Set EPI (no anda)')
        self.tirfButton = QtGui.QPushButton('TIRF (no anda)')
        self.tirfButton.setCheckable(True)
        self.epiButton = QtGui.QPushButton('EPI (no anda)')
        self.epiButton.setCheckable(True)
        self.stagePosLabel = QtGui.QLabel('0 mm')
        self.getIntButton = QtGui.QPushButton('Get intensities')
        self.getIntButton.clicked.connect(self.getIntensities)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.redControl, 0, 1)
        grid.addWidget(self.blueControl, 0, 0)
        grid.addWidget(self.greenControl, 0, 2)
        grid.addWidget(self.findTirfButton, 1, 0)
        grid.addWidget(self.setEpiButton, 2, 0)
        grid.addWidget(self.tirfButton, 1, 1)
        grid.addWidget(self.epiButton, 2, 1)
        grid.addWidget(self.stagePosLabel, 1, 2)
        grid.addWidget(self.getIntButton, 2, 2)

        # Current power update routine
        self.updatePowers = UpdatePowers(self)
        self.updateThread = QtCore.QThread()
        self.updatePowers.moveToThread(self.updateThread)
        self.updateThread.start()
        self.updateThread.started.connect(self.updatePowers.update)

        # Intensity measurement in separate thread to keep the GUI responsive
        self.worker = IntensityWorker(self, 20, 3, self.calibrationPath)
        self.worker.updateSignal.connect(self.updateIntensities)
        self.intensityThread = QtCore.QThread()
        self.worker.moveToThread(self.intensityThread)
        self.intensityThread.started.connect(self.worker.start)

    def getIntensities(self):
        # Flip measurement mirror
        self.main.flipperInPath(True)
        time.sleep(0.5)
        self.daq.digital_IO[3] = False
        self.intensityThread.start()

    def updateIntensities(self, data):
        for d in data:
            for c in self.controls:
                if int(c.name.text()[4:7]) == d['laser']:
                    c.intensityEdit.setText(str(np.round(d['intensity'], 1)))
                    c.calibratedCheck.setChecked(d['calibrated'])
                    c.voltageLabel.setText(str(np.round(d['voltage'], 2)))
        self.intensityThread.quit()

        # Flip measurement mirror back
        self.daq.digital_IO[3] = True
        self.main.flipperInPath(False)

    def closeShutters(self):
        for control in self.shuttLasers:
            control.shutterBox.setChecked(False)

    def closeEvent(self, *args, **kwargs):
        self.closeShutters()
        self.updateThread.quit()
        super().closeEvent(*args, **kwargs)


class IntensityWorker(QtCore.QObject):

    # This signal carries the photodiode's measured voltages
    updateSignal = QtCore.pyqtSignal(np.ndarray)
    doneSignal = QtCore.pyqtSignal()

    def __init__(self, main, scansPerS, port, calPath, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = main
        self.scansPerS = scansPerS
        self.port = port

        cal_dt = [('laser', int), ('p0', float), ('p1', float)]
        self.calibration = np.loadtxt(calPath, dtype=cal_dt, skiprows=1)

    def start(self):
        self.stream = daqStream(self.main.daq, self.scansPerS, self.port)
        self.stream.start()

        shuttLasers = self.main.shuttLasers
        enabledLasers = [control.laser.enabled for control in shuttLasers]
        enabledLasers = np.array(enabledLasers, dtype=bool)
        enabledControls = shuttLasers[enabledLasers]

        # Results signal
        dt = np.dtype([('laser', int),
                       ('intensity', float),
                       ('calibrated', bool),
                       ('voltage', float)])
        signal = np.zeros(len(enabledControls), dtype=dt)

        # Record shutters state
        shutterState = [ctl.shutterBox.isChecked() for ctl in enabledControls]

        j = 0
        # Measure each laser intensity
        for control in enabledControls:
            others = [ctrl for ctrl in enabledControls if ctrl != control]
            for ctrl in others:
                ctrl.shutterBox.setChecked(False)
            control.shutterBox.setChecked(True)

            # Initial intensity measurement
            trace = np.zeros(20)
            for i in np.arange(len(trace)):
                trace[i] = self.stream.getNewData()

            # Wait until intensity fluctuations are below 0.5%
            mean = np.mean(trace)
            dev = np.std(trace)
            it = 0
            while dev / mean > 0.005 and it < 100:
                trace[:-1] = trace[1:]
                trace[-1] = self.stream.getNewData()
                mean = np.mean(trace)
                dev = np.std(trace)
                it += 1

            # Store intensity measurement using calibrated voltages
            laser = int(control.name.text()[4:7])
            row = np.where(self.calibration['laser'] == laser)[0][0]
            calibration = self.calibration[row]
            intensity = calibration['p0'] + mean * calibration['p1']
            calibrated = calibration['p1'] != 1
            signal[j] = (laser, intensity, calibrated, mean)
            j += 1

        # Stop DAQ streaming
        self.stream.stop()

        # Load original shutter state
        i = 0
        for control in enabledControls:
            control.shutterBox.setChecked(shutterState[i])
            i += 1

        self.updateSignal.emit(signal)
        self.doneSignal.emit()


class LaserControl(QtGui.QFrame):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)

        # Power widget
        self.setPointLabel = QtGui.QLabel('Setpoint')
        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))
        self.setPointEdit.setFixedWidth(50)
        self.setPointEdit.setAlignment(QtCore.Qt.AlignRight)
        self.mWLabel = QtGui.QLabel('mW')

        self.powerLabel = QtGui.QLabel('Power')
        powerMag = self.laser.power.magnitude
        self.powerIndicator = QtGui.QLabel(str(powerMag))
        self.powerIndicator.setFixedWidth(50)
        self.powerIndicator.setAlignment(QtCore.Qt.AlignRight)
        self.mWLabel2 = QtGui.QLabel('mW')

        self.intensityLabel = QtGui.QLabel('Intensity')
        self.intensityEdit = QtGui.QLabel('0')
        self.intensityEdit.setAlignment(QtCore.Qt.AlignRight)
        self.kWcm2Label = QtGui.QLabel('kW/cm^2')
        self.voltageLabel = QtGui.QLabel('0')
        self.voltageLabel.setAlignment(QtCore.Qt.AlignRight)
        self.VLabel = QtGui.QLabel('V')

        self.calibratedCheck = QtGui.QCheckBox('Calibrated')

        powerWidget = QtGui.QWidget()
        powerGrid = QtGui.QGridLayout()
        powerWidget.setLayout(powerGrid)
        powerGrid.addWidget(self.setPointLabel, 0, 0, 1, 2)
        powerGrid.addWidget(self.setPointEdit, 1, 0)
        powerGrid.addWidget(self.mWLabel, 1, 1)
        powerGrid.addWidget(self.powerLabel, 2, 0, 1, 2)
        powerGrid.addWidget(self.powerIndicator, 3, 0)
        powerGrid.addWidget(self.mWLabel2, 3, 1)
        powerGrid.addWidget(self.intensityLabel, 4, 0, 1, 2)
        powerGrid.addWidget(self.intensityEdit, 5, 0)
        powerGrid.addWidget(self.kWcm2Label, 5, 1)
        powerGrid.addWidget(self.voltageLabel, 6, 0)
        powerGrid.addWidget(self.VLabel, 6, 1)
        powerGrid.addWidget(self.calibratedCheck, 7, 0, 1, 2)

        # Shutter port
        if self.port is not None:
            self.shutterBox = QtGui.QCheckBox('Shutter open')
            powerGrid.addWidget(self.shutterBox, 8, 0, 1, 2)
            self.shutterBox.stateChanged.connect(self.shutterAction)

            if invert:
                self.daq.digital_IO[self.port] = True
                self.states = {2: False, 0: True}
            else:
                self.daq.digital_IO[self.port] = False
                self.states = {2: True, 0: False}

        # Slider
        self.maxpower = QtGui.QLabel(str(prange[1]))
        self.maxpower.setAlignment(QtCore.Qt.AlignCenter)
        self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setMinimum(prange[0])
        self.slider.setMaximum(prange[1])
        self.slider.setTickInterval(tickInterval)
        self.slider.setSingleStep(singleStep)
        self.slider.setValue(self.laser.power.magnitude)
        self.minpower = QtGui.QLabel(str(prange[0]))
        self.minpower.setAlignment(QtCore.Qt.AlignCenter)

        # ON/OFF button
        self.enableButton = QtGui.QPushButton('ON')
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
        if self.laser.enabled:
            self.enableButton.setChecked(True)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.name, 0, 0, 1, 2)
        grid.addWidget(powerWidget, 3, 0)
        grid.addWidget(self.maxpower, 1, 1)
        grid.addWidget(self.slider, 2, 1, 5, 1)
        grid.addWidget(self.minpower, 7, 1)
        grid.addWidget(self.enableButton, 8, 0, 1, 2)

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)
        self.slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)
        self.powerChanged = True

    def shutterAction(self, state):
        self.daq.digital_IO[self.port] = self.states[state]

    def toggleLaser(self):
        if self.enableButton.isChecked():
            self.laser.enabled = True
            self.powerChanged = True
        else:
            self.laser.enabled = False

    def enableLaser(self):
        self.laser.enabled = True
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW
        self.powerChanged = True

    def changeSlider(self, value):
        self.laser.power_sp = self.slider.value() * self.mW
        self.setPointEdit.setText(str(self.laser.power_sp.magnitude))
        self.powerChanged = True

    def changeEdit(self):
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW
        self.slider.setValue(self.laser.power_sp.magnitude)
        self.powerChanged = True

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)


if __name__ == '__main__':

    app = QtGui.QApplication([])

    import tormenta.control.instruments as instruments
    blueDriver = 'rgblasersystems.minilasevo.MiniLasEvo'
    greenDriver = 'laserquantum.ventus.Ventus'

    with instruments.Laser('mpb.vfl.VFL', 'COM11') as redlaser, \
            instruments.Laser(blueDriver, 'COM7') as bluelaser, \
            instruments.Laser(greenDriver, 'COM13') as greenlaser, \
            instruments.DAQ() as daq:

        print(redlaser.idn, bluelaser.idn, greenlaser.idn, daq.idn)
        win = LaserWidget((redlaser, bluelaser, greenlaser), daq)
        win.show()

        app.exec_()
