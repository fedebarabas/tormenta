# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Federico Barabas, Luciano Masullo
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import pygame

from lantz import Q_

from instruments import ScanZ, Webcam   # , DAQ
from pi import PI


class FocusWidget(QtGui.QFrame):

#    def __init__(self, DAQ, scanZ, main=None, *args, **kwargs):
    def __init__(self, scanZ, main=None, *args, **kwargs):

        super(FocusWidget, self).__init__(*args, **kwargs)


        self.main = main  # main va a ser RecordingWidget de control.py
#        self.DAQ = DAQ
#        try:
#            self.DAQ.streamStop()
#        except:
#            pass

        self.webcam = Webcam()

        self.z = scanZ
        self.setPoint = 0
        self.calibrationResult = [0, 0]

        self.z.hostPosition = 'left'

        self.V = Q_(1, 'V')
        self.um = Q_(1, 'um')
        self.nm = Q_(1, 'nm')

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Thread for getting data from DAQ
        self.scansPerS = 10
#        self.scansPerS = 20
#        self.stream = daqStream(DAQ, scansPerS)
#        self.streamThread = QtCore.QThread()
#        self.stream.moveToThread(self.streamThread)
#        self.streamThread.started.connect(self.stream.start)
#        self.streamThread.start()

        self.focusCalib = focusCalibration(self)
        self.focusCalibThread = QtCore.QThread()
        self.focusCalib.moveToThread(self.focusCalibThread)
        self.focusCalibButton = QtGui.QPushButton('Calibrate')
        self.focusCalibButton.clicked.connect(self.focusCalib.start)
        self.focusCalibThread.start()

        try:
            prevCal = np.around(np.loadtxt('calibration')[0]/10)
            text = '0,1 mV --> {} nm'.format(prevCal)
            self.calibrationDisplay = QtGui.QLineEdit(text)
        except:
            self.calibrationDisplay = QtGui.QLineEdit('0 mV --> 0 nm')

        self.calibrationDisplay.setReadOnly(False)

        # Focus lock widgets
        self.kpEdit = QtGui.QLineEdit('0.005')
        self.kpEdit.textChanged.connect(self.unlockFocus)
        self.kpLabel = QtGui.QLabel('kp')
        self.kiEdit = QtGui.QLineEdit('0.0001')
        self.kiEdit.textChanged.connect(self.unlockFocus)
        self.kiLabel = QtGui.QLabel('ki')
        self.lockButton = QtGui.QPushButton('Lock')
        self.lockButton.setCheckable(True)
        self.lockButton.clicked.connect(self.toggleFocus)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)

        self.focusDataBox = QtGui.QCheckBox('Save focus data')

#        self.exportDataButton = QtGui.QPushButton('Export data')
#        self.exportDataButton.clicked.connect(self.exportData)
#        self.focusAnalisisButton = QtGui.QPushButton('Focus analisis')
#        self.focusAnalisisButton.clicked.connect(self.analizeFocus)
        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')
#        self.focusPropertiesDisplay.setReadOnly(True)

#        style = QtGui.QFrame.Panel | QtGui.QFrame.Raised
#        self.focusPropertiesDisplay.setFrameStyle(style)


        self.webcamView = webcamView(self.webcam)
        self.graph = FocusLockGraph(self, main)

        self.focusTime = 1000 / self.scansPerS
        self.focusTimer = QtCore.QTimer()
        self.focusTimer.timeout.connect(self.update)
        self.focusTimer.start(self.focusTime)

        self.locked = False
        self.n = 1
        self.max_dev = 0

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.graph, 0, 0, 1, 3)
        grid.addWidget(self.webcamView, 0, 3, 1, 3)
        grid.addWidget(self.focusCalibButton, 1, 0)
        grid.addWidget(self.calibrationDisplay, 2, 0)
        grid.addWidget(self.kpLabel, 1, 3)
        grid.addWidget(self.kpEdit, 1, 4)
        grid.addWidget(self.kiLabel, 2, 3)
        grid.addWidget(self.kiEdit, 2, 4)
        grid.addWidget(self.lockButton, 1, 5, 2, 1)
        grid.addWidget(self.focusDataBox, 1, 2)

        grid.setColumnMinimumWidth(1, 100)
#        grid.setColumnMinimumWidth(2, 40)
        grid.setColumnMinimumWidth(0, 100)

    def update(self):
        self.webcamView.update()
        self.graph.update()

        if self.locked:
            self.updatePI()

    def toggleFocus(self):
        if self.lockButton.isChecked():
            self.setPoint = self.webcamView.focusSignal
            self.graph.line = self.graph.plot.addLine(y=self.setPoint, pen='r')
            self.PI = PI(self.setPoint,
                         np.float(self.kpEdit.text()),
                         np.float(self.kiEdit.text()))

            self.initialZ = self.z.position
            self.locked = True

        else:
            self.unlockFocus()

    def unlockFocus(self):
        if self.locked:
            self.locked = False
            self.lockButton.setChecked(False)
            self.graph.plot.removeItem(self.graph.line)

    def updatePI(self):

        # TODO: explain ifs
        self.distance = self.z.position - self.initialZ
#        out = self.PI.update(self.stream.newData)
        out = self.PI.update(self.webcamView.focusSignal)
        if abs(self.distance) > 10 * self.um or abs(out) > 5:
            self.unlockFocus()
        else:
            self.z.moveRelative(out * self.um)

    def moveZ(self, value):
        self.z.position = value

    def exportData(self):

        self.sizeofData = np.size(self.graph.savedDataSignal)
        self.savedData = [np.ones(self.sizeofData)*self.setPoint,
                          self.graph.savedDataSignal, self.graph.savedDataTime]
        np.savetxt('{}_focusdata'.format(self.main.filename()), self.savedData)
        self.graph.savedDataSignal = []
        self.graph.savedDataTime = []

#        self.plot = plt.plot(self.graph.savedDataTime,
#                             self.graph.savedDataSignal, 'b-',
#                             self.graph.savedDataTime,
#                             np.ones(self.sizeofData)*self.setPoint, 'r-')

#        self.graph.savedDataPosition = []

    def analizeFocus(self):

        if self.n == 1:
            self.mean = self.webcamView.focusSignal
            self.mean2 = self.webcamView.focusSignal**2
        else:
            self.mean += (self.webcamView.focusSignal - self.mean)/self.n
            self.mean2 += (self.webcamView.focusSignal**2 - self.mean2)/self.n

        self.std = np.sqrt(self.mean2 - self.mean**2)

        self.max_dev = np.max([self.max_dev,
                              self.webcamView.focusSignal - self.setPoint])

        statData = 'std = {}    max_dev = {}'.format(np.round(self.std, 3),
                                                     np.round(self.max_dev, 3))
        self.graph.statistics.setText(statData)

        self.n += 1

    def closeEvent(self, *args, **kwargs):

        self.focusTimer.stop()
#        if self.lockButton.isChecked():
#            self.lockTimer.stop()

#        self.DAQ.streamStop()
#        self.streamThread.terminate()

        self.webcam.stop()

        super().closeEvent(*args, **kwargs)


class webcamView(pg.GraphicsLayoutWidget):

    def __init__(self, webcam, *args, **kwargs):

        super(webcamView, self).__init__(*args, **kwargs)
        self.webcam = webcam
        image = self.webcam.get_image()
        self.sensorSize = pygame.surfarray.array2d(image).shape

        self.img = pg.ImageItem(border='w')
        self.view = self.addViewBox()
        self.view.setAspectLocked(True)  # square pixels
        self.view.addItem(self.img)

        # TODO: vale la pena promediar?
        # TODO: potencia óptima del láser
        # TODO: caja
        # TODO: circuito

    def update(self):

        runs = 1
        imageArray = np.zeros((runs, self.sensorSize[0], self.sensorSize[1]),
                              np.float)

        # mucha CPU
        for i in range(runs):
            image = self.webcam.get_image()
            image = pygame.surfarray.array2d(image).astype(np.float)
            imageArray[i] = image / np.sum(image)

        finalImage = np.sum(imageArray, 0)
        self.img.setImage(finalImage)
        self.focusSignal = (ndi.measurements.center_of_mass(finalImage)[0] -
                            self.sensorSize[0] / 2)


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, focusWidget, main=None, *args, **kwargs):

        super(FocusLockGraph, self).__init__(*args, **kwargs)

        self.focusWidget = focusWidget
        self.main = main
        self.scansPerS = self.focusWidget.scansPerS
        self.analize = self.focusWidget.analizeFocus
        self.focusDataBox = self.focusWidget.focusDataBox
        self.savedDataSignal = []
        self.savedDataTime = []
        self.savedDataPosition = []

        self.setWindowTitle('Focus')
        self.setAntialiasing(True)

        self.npoints = 200
        self.data = np.zeros(self.npoints)
        self.ptr = 0

        # Graph without a fixed range
        self.statistics = pg.LabelItem(justify='right')
        self.addItem(self.statistics)
        self.statistics.setText('---')
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabels(bottom=('Tiempo', 's'),
                            left=('Laser position', 'px'))
        self.plot.showGrid(x=True, y=True)
        self.focusCurve = self.plot.plot(pen='y')

        self.time = np.zeros(self.npoints)
        self.startTime = ptime.time()

        if self.main is not None:
            self.recButton = self.main.recButton

    def update(self):
        """ Update the data displayed in the graphs
        """
        self.focusSignal = self.focusWidget.webcamView.focusSignal

        if self.ptr < self.npoints:
            self.data[self.ptr] = self.focusSignal
            self.time[self.ptr] = ptime.time() - self.startTime
            self.focusCurve.setData(self.time[1:self.ptr + 1],
                                    self.data[1:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.focusSignal
            self.time[:-1] = self.time[1:]
            self.time[-1] = ptime.time() - self.startTime

            self.focusCurve.setData(self.time, self.data)

        self.ptr += 1

        if self.main is not None:

            if self.recButton.isChecked():
                self.savedDataSignal.append(self.focusSignal)
                self.savedDataTime.append(self.time[-1])
#               self.savedDataPosition.append(self.DAQ.position)

            if self.recButton.isChecked():
                self.analize()


class focusCalibration(QtCore.QObject):

    def __init__(self, mainwidget, *args, **kwargs):

        super(focusCalibration, self).__init__(*args, **kwargs)
        self.signalData = []
        self.positionData = []
        self.nm = Q_(1, 'nm')
        self.step = 50 * self.nm
#        self.stream = mainwidget.stream
        self.z = mainwidget.z
        self.mainwidget = mainwidget  # mainwidget será FocusLockWidget

    def start(self):

        for i in range(200):

            self.signalData.append(self.stream.newData)
            self.positionData.append(self.z.position.magnitude)
            self.z.moveRelative(self.step)
            time.sleep(0.5)

        self.argmax = np.argmax(self.signalData)
        self.argmin = np.argmin(self.signalData)
        self.signalData = self.signalData[self.argmin:self.argmax]
        self.positionData = self.positionData[self.argmin:self.argmax]

        poly = np.polyfit(np.array(self.signalData),
                          np.array(self.positionData), 1)
        self.calibrationResult = np.around(poly, 2)
        self.export()

    def export(self):

        np.savetxt('calibration', self.calibrationResult)
        cal = np.around(np.abs(self.calibrationResult[0])*0.1, 1)
        calText = '0,1 mV --> {} nm'.format(cal)
        self.mainwidget.calibrationDisplay.setText(calText)


class daqStream(QtCore.QObject):
    """This stream only takes care of getting data from the Labjack device."""

    def __init__(self, DAQ, scansPerS, *args, **kwargs):

        super(daqStream, self).__init__(*args, **kwargs)

        self.DAQ = DAQ
        self.scansPerS = scansPerS
        self.port = 'AIN0'
        names = [self.port + "_NEGATIVE_CH", self.port + "_RANGE",
                 "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
        # single-ended, +/-1V, 0, 0 (defaults)
        # Voltage Ranges: ±10V, ±1V, ±0.1V, and ±0.01V
        values = [self.DAQ.constants.GND, 0.1, 0, 0]
        self.DAQ.writeNames(names, values)
        self.newData = 0.0

    def start(self):
        scanRate = 5000
        scansPerRead = int(scanRate/self.scansPerS)
        portAddress = self.DAQ.address(self.port)[0]
        scanRate = self.DAQ.streamStart(scansPerRead, [portAddress], scanRate)
        self.newData = np.mean(self.DAQ.streamRead()[0])
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1)

    def stop(self):
        pass
        # TODO: stop

    def update(self):
        self.newData = np.mean(self.DAQ.streamRead()[0])

if __name__ == '__main__':

    app = QtGui.QApplication([])

#    with DAQ() as DAQ, ScanZ(12) as z:
    with ScanZ(12) as z:

        win = FocusWidget(z)
#        win = FocusWidget(DAQ, z)
        win.show()

        app.exec_()
