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

        bluePower = np.round(self.widget.bluelaser.power.magnitude, 1)
        self.widget.blueControl.powerIndicator.setText(str(bluePower))

        greenPower = np.round(self.widget.greenlaser.power.magnitude, 1)
        greenPSUTemp = np.round(self.widget.greenlaser.psuTemp.magnitude, 1)
        self.widget.greenControl.powerIndicator.setText(str(greenPower))
        self.widget.greenControl.psuTempInd.setText(str(greenPSUTemp))

        redPower = np.round(self.widget.redlaser.power.magnitude, 1)
        self.widget.redControl.powerIndicator.setText(str(redPower))

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

        # Blue laser control widget
        self.blueControl = LaserControl(self.bluelaser, '<h3>405nm</h3>',
                                        color=(73, 0, 188), prange=(0, 53),
                                        tickInterval=5, singleStep=0.1)
        # Green laser control widget
        self.greenControl = LaserControl(self.greenlaser, '<h3>532nm</h3>',
                                         color=(80, 255, 0), prange=(0, 1500),
                                         tickInterval=10, singleStep=1,
                                         daq=self.daq, port=0, invert=False)

        # Additional green laser temperature indicators
        self.greenControl.tempFrame = QtGui.QFrame(self.greenControl)
        self.greenControl.tempFrame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Plain)
        greenTempGrid = QtGui.QGridLayout()
        self.greenControl.tempFrame.setLayout(greenTempGrid)
        self.greenControl.psuTempInd = QtGui.QLabel('0')
        self.greenControl.psuTempInd.setAlignment(QtCore.Qt.AlignRight)
        self.greenControl.laserTempInd = QtGui.QLabel('0')
        self.greenControl.laserTempInd.setAlignment(QtCore.Qt.AlignRight)
        greenTempGrid.addWidget(QtGui.QLabel('Temperature PSU'), 0, 0)
        greenTempGrid.addWidget(self.greenControl.psuTempInd, 0, 1)
        greenTempGrid.addWidget(QtGui.QLabel('ºC'), 0, 2)
        greenTempGrid.addWidget(QtGui.QLabel('Temperature Laser'), 1, 0)
        greenTempGrid.addWidget(self.greenControl.laserTempInd, 1, 1)
        greenTempGrid.addWidget(QtGui.QLabel('ºC'), 1, 2)
        self.greenControl.grid.addWidget(self.greenControl.tempFrame, 2, 0, 1, 2)

        # Red laser control widget
        self.redControl = LaserControl(self.redlaser, '<h3>642nm</h3>',
                                       color=(255, 11, 0), prange=(150, 1500),
                                       tickInterval=100, singleStep=10,
                                       daq=self.daq, port=1)

        # Additional red laser temperature indicators
        self.redControl.tempFrame = QtGui.QFrame(self.redControl)
        self.redControl.tempFrame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Plain)
        redTempGrid = QtGui.QGridLayout()
        self.redControl.tempFrame.setLayout(redTempGrid)
        self.redControl.psuTempInd = QtGui.QLabel('0')
        self.redControl.psuTempInd.setAlignment(QtCore.Qt.AlignRight)
        self.redControl.laserTempInd = QtGui.QLabel('0')
        self.redControl.laserTempInd.setAlignment(QtCore.Qt.AlignRight)
        redTempGrid.addWidget(QtGui.QLabel('Temperature SHG'), 0, 0)
        redTempGrid.addWidget(self.greenControl.psuTempInd, 0, 1)
        redTempGrid.addWidget(QtGui.QLabel('ºC'), 0, 2)
        redTempGrid.addWidget(QtGui.QLabel('Temperature Laser'), 1, 0)
        redTempGrid.addWidget(self.greenControl.laserTempInd, 1, 1)
        redTempGrid.addWidget(QtGui.QLabel('ºC'), 1, 2)
        self.redControl.grid.addWidget(self.redControl.tempFrame, 2, 0, 1, 2)

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
        grid.addWidget(self.blueControl, 0, 0)
        grid.addWidget(self.greenControl, 0, 1)
        grid.addWidget(self.redControl, 0, 2)
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
        # Get initial flipper mirror state and put it on path
        self.flipState = not(self.main.flipperButton.isChecked())
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

        # Put flipper in initial state
        self.daq.digital_IO[3] = True
        self.main.flipperInPath(self.flipState)

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
        try:
            self.calibration = np.loadtxt(calPath, dtype=cal_dt, skiprows=1)
        except:
            self.calibration = np.zeros(3, dtype=cal_dt)
            self.calibration['laser'] = [405, 532, 640]

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
        signal = np.zeros(len(shuttLasers) + 1, dtype=dt)

        # Record shutters state
        shutterState = [ctl.shutterBox.isChecked() for ctl in shuttLasers]

        # Measure each laser intensity
        row = np.where(self.calibration['laser'] == 405)[0][0]
        calibration = self.calibration[row]
        power = self.main.bluelaser.power.magnitude
        intensity = calibration['p0'] + power*calibration['p1']
        calibrated = calibration['p1'] != 1
        signal[0] = (405, intensity, calibrated, power)

        j = 1
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
            intensity = calibration['p0'] + mean*calibration['p1']
            calibrated = calibration['p1'] != 1
            signal[j] = (laser, intensity, calibrated, mean)
            j += 1

        # Stop DAQ streaming
        self.stream.stop()

        # Load original shutter state
        i = 0
        for control in shuttLasers:
            control.shutterBox.setChecked(shutterState[i])
            i += 1

        self.updateSignal.emit(signal)
        self.doneSignal.emit()


class LaserControl(QtGui.QFrame):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 daq=None, port=None, invert=True, addArgs=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')
        self.daq = daq
        self.port = port

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)
        self.name.setAlignment(QtCore.Qt.AlignCenter)
        self.name.setStyleSheet("font-size:16px")
        self.name.setFixedHeight(40)

        # Power widget
        self.setPointLabel = QtGui.QLabel('Setpoint')
        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))
        self.setPointEdit.setFixedWidth(50)
        self.setPointEdit.setAlignment(QtCore.Qt.AlignRight)

        self.powerLabel = QtGui.QLabel('Power')
        powerMag = self.laser.power.magnitude
        self.powerIndicator = QtGui.QLabel(str(powerMag))
        self.powerIndicator.setFixedWidth(50)
        self.powerIndicator.setAlignment(QtCore.Qt.AlignRight)

        self.intensityLabel = QtGui.QLabel('Intensity')
        self.intensityEdit = QtGui.QLabel('0')
        self.intensityEdit.setAlignment(QtCore.Qt.AlignRight)
        self.voltageLabel = QtGui.QLabel('0')
        self.voltageLabel.setAlignment(QtCore.Qt.AlignRight)

        self.calibratedCheck = QtGui.QCheckBox('Calibrated')

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

        powerFrame = QtGui.QFrame(self)
        self.powerGrid = QtGui.QGridLayout()
        powerFrame.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Plain)
        powerFrame.setLayout(self.powerGrid)
        self.powerGrid.addWidget(self.setPointLabel, 1, 0, 1, 2)
        self.powerGrid.addWidget(self.setPointEdit, 2, 0)
        self.powerGrid.addWidget(QtGui.QLabel('mW'), 2, 1)
        self.powerGrid.addWidget(self.powerLabel, 3, 0, 1, 2)
        self.powerGrid.addWidget(self.powerIndicator, 4, 0)
        self.powerGrid.addWidget(QtGui.QLabel('mW'), 4, 1)
        self.powerGrid.addWidget(self.intensityLabel, 5, 0, 1, 2)
        self.powerGrid.addWidget(self.intensityEdit, 6, 0)
        self.powerGrid.addWidget(QtGui.QLabel('kW/cm^2'), 5, 1)
        self.powerGrid.addWidget(self.voltageLabel, 7, 0)
        self.powerGrid.addWidget(QtGui.QLabel('V'), 7, 1)
        self.powerGrid.addWidget(self.calibratedCheck, 8, 0, 1, 2)
        self.powerGrid.addWidget(self.maxpower, 0, 3)
        self.powerGrid.addWidget(self.slider, 1, 3, 8, 1)
        self.powerGrid.addWidget(self.minpower, 9, 3)

        # ON/OFF button
        self.enableButton = QtGui.QPushButton('ON')
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
        if self.laser.enabled:
            self.enableButton.setChecked(True)

        self.grid = QtGui.QGridLayout()
        self.setLayout(self.grid)
        self.grid.addWidget(self.name, 0, 0, 1, 2)
        self.grid.addWidget(powerFrame, 1, 0, 1, 2)

        # Shutter port
        if self.port is None:
            self.grid.addWidget(self.enableButton, 8, 0, 1, 2)
        else:
            self.shutterBox = QtGui.QCheckBox('Open')
            self.shutterBox.setFixedWidth(50)
            self.shutterBox.stateChanged.connect(self.shutterAction)
            self.grid.addWidget(self.enableButton, 8, 0)
            self.grid.addWidget(self.shutterBox, 8, 1)

            if invert:
                self.daq.digital_IO[self.port] = True
                self.states = {2: False, 0: True}
            else:
                self.daq.digital_IO[self.port] = False
                self.states = {2: True, 0: False}

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

    with instruments.Laser('mpb.vfl.VFL', 'COM3') as redlaser, \
            instruments.Laser(blueDriver, 'COM7') as bluelaser, \
            instruments.Laser(greenDriver, 'COM13') as greenlaser, \
            instruments.DAQ() as daq:

        print(redlaser.idn, bluelaser.idn, greenlaser.idn, daq.idn)
        win = QtGui.QMainWindow()
        win.setWindowTitle('Laser control')
        laserWidget = LaserWidget(None, (redlaser, bluelaser, greenlaser), daq)
        win.setCentralWidget(laserWidget)
        win.show()

        app.exec_()
