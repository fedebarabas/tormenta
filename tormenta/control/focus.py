# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@authors: Federico Barabas, Luciano Masullo
"""

import numpy as np
import time
import scipy.ndimage as ndi

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import Dock, DockArea
import pyqtgraph.ptime as ptime
import pygame

from lantz import Q_

import tormenta.control.instruments as instruments
import tormenta.control.pi as pi


class FocusWidget(QtGui.QFrame):

    def __init__(self, scanZ, main=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.setMinimumSize(2, 350)

        self.main = main  # main va a ser RecordingWidget de control.py

        self.webcam = instruments.Webcam()

        self.z = scanZ
        self.setPoint = 0
        self.calibrationResult = [0, 0]

        self.z.hostPosition = 'left'

        self.V = Q_(1, 'V')
        self.um = Q_(1, 'um')
        self.nm = Q_(1, 'nm')

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.scansPerS = 10
        self.ProcessData = ProcessData(self.webcam)

        # Focus lock widgets
        self.kpEdit = QtGui.QLineEdit('4')
        self.kpEdit.textChanged.connect(self.unlockFocus)
        self.kpLabel = QtGui.QLabel('kp')
        self.kiEdit = QtGui.QLineEdit('0.004')
        self.kiEdit.textChanged.connect(self.unlockFocus)
        self.kiLabel = QtGui.QLabel('ki')
        self.lockButton = QtGui.QPushButton('Lock')
        self.lockButton.setCheckable(True)
        self.lockButton.clicked.connect(self.toggleFocus)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)

        self.focusDataBox = QtGui.QCheckBox('Save focus data')
        self.focusPropertiesDisplay = QtGui.QLabel(' st_dev = 0  max_dev = 0')

        self.graph = FocusLockGraph(self, main)

        self.focusTime = 1000 / self.scansPerS
        self.focusTimer = QtCore.QTimer()
        self.focusTimer.timeout.connect(self.update)
        self.focusTimer.start(self.focusTime)

        self.locked = False
        self.n = 1
        self.max_dev = 0

        self.focusCalib = FocusCalibration(self)
        self.focusCalibThread = QtCore.QThread()
        self.focusCalib.moveToThread(self.focusCalibThread)
        self.focusCalibButton = QtGui.QPushButton('Calibrate')
        self.focusCalibButton.clicked.connect(self.focusCalib.start)
        self.focusCalibThread.start()

        try:
            prevCal = np.around(np.loadtxt('calibration')[0]/10)
            text = '1 px --> {} nm'.format(prevCal)
            self.calibrationDisplay = QtGui.QLineEdit(text)
        except:
            self.calibrationDisplay = QtGui.QLineEdit('0 px --> 0 nm')

        self.calibrationDisplay.setReadOnly(False)

        self.webcamgraph = WebcamGraph(self)

        dockArea = DockArea()
        graphDock = Dock("Laser position", size=(400, 200))
        graphDock.addWidget(self.graph)
        dockArea.addDock(graphDock)
        webcamDock = Dock("Webcam view", size=(200, 200))
        webcamDock.addWidget(self.webcamgraph)
        dockArea.addDock(webcamDock, 'right', graphDock)

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(dockArea, 0, 0, 1, 6)
        grid.addWidget(self.focusCalibButton, 1, 0)
        grid.addWidget(self.calibrationDisplay, 2, 0)
        grid.addWidget(self.kpLabel, 1, 3)
        grid.addWidget(self.kpEdit, 1, 4)
        grid.addWidget(self.kiLabel, 2, 3)
        grid.addWidget(self.kiEdit, 2, 4)
        grid.addWidget(self.lockButton, 1, 5, 2, 1)
        grid.addWidget(self.focusDataBox, 1, 2)

    def update(self):
        try:
            self.ProcessData.update()
            self.graph.update()
            self.webcamgraph.update()
        except:
            pass

        if self.locked:
            self.updatePI()

    def toggleFocus(self):
        if self.lockButton.isChecked():
            self.setPoint = self.ProcessData.focusSignal
            self.graph.line = self.graph.plot.addLine(y=self.setPoint, pen='r')
            self.PI = pi.PI(self.setPoint, 0.001, np.float(self.kpEdit.text()),
                            np.float(self.kiEdit.text()))

            self.lockN = 1
            self.lockMean = self.setPoint
            self.graph.setLine = self.graph.plot.addLine(y=self.lockMean,
                                                         pen='c')
            self.initialZ = self.z.position
            self.locked = True

        else:
            self.unlockFocus()

    def unlockFocus(self):
        if self.locked:
            self.locked = False
            self.lockButton.setChecked(False)
            self.graph.plot.removeItem(self.graph.line)
            self.graph.plot.removeItem(self.graph.setLine)

    def updatePI(self):

        # Safety unlocking
        self.distance = self.z.position - self.initialZ
        cm = self.ProcessData.focusSignal
        out = self.PI.update(cm)

        self.lockN += 1
        self.lockMean += (cm - self.lockMean)/(self.lockN + 1)
        self.graph.setLine.setValue(self.lockMean)

        if abs(self.distance) > 10 * self.um or abs(out) > 5:
            self.unlockFocus()
        else:
            self.z.moveRelative(out * self.um)

    def exportData(self):

        self.sizeofData = np.size(self.graph.savedDataSignal)
        self.savedData = [np.ones(self.sizeofData)*self.setPoint,
                          self.graph.savedDataSignal, self.graph.savedDataTime]
        np.savetxt('{}_focusdata'.format(self.main.filename()), self.savedData)
        self.graph.savedDataSignal = []
        self.graph.savedDataTime = []

    def analizeFocus(self):

        if self.n == 1:
            self.mean = self.ProcessData.focusSignal
            self.mean2 = self.ProcessData.focusSignal**2
        else:
            self.mean += (self.ProcessData.focusSignal - self.mean)/self.n
            self.mean2 += (self.ProcessData.focusSignal**2 - self.mean2)/self.n

        # Stats
        self.std = np.sqrt(self.mean2 - self.mean**2)
        self.max_dev = np.max([self.max_dev,
                              self.ProcessData.focusSignal - self.setPoint])
        statData = 'std = {}    max_dev = {}'.format(np.round(self.std, 3),
                                                     np.round(self.max_dev, 3))
        self.graph.statistics.setText(statData)

        self.n += 1

    def closeEvent(self, *args, **kwargs):
        self.focusTimer.stop()
        self.webcam.stop()
        super().closeEvent(*args, **kwargs)


class ProcessData(pg.GraphicsLayoutWidget):

    def __init__(self, webcam, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.webcam = webcam
        image = self.webcam.get_image()
        self.sensorSize = np.array(pygame.surfarray.array2d(image).shape)
        self.focusSignal = 0

    def update(self):

        runs = 1
        imageArray = np.zeros((runs, self.sensorSize[0], self.sensorSize[1]),
                              np.float)
        # mucha CPU
        for i in range(runs):
            image = self.webcam.get_image()
            image = pygame.surfarray.array2d(image).astype(np.float)
            try:
                imageArray[i] = image / np.sum(image)
            except:
                print(np.sum(image))

        finalImage = np.sum(imageArray, 0)
        self.massCenter = np.array(ndi.measurements.center_of_mass(finalImage))
        self.massCenter[0] = self.massCenter[0] - self.sensorSize[0] / 2
        self.massCenter[1] = self.massCenter[1] - self.sensorSize[1] / 2
        self.focusSignal = self.massCenter[0]

        self.image = finalImage


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, focusWidget, main=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.focusWidget = focusWidget
        self.main = main
        self.analize = self.focusWidget.analizeFocus
        self.focusDataBox = self.focusWidget.focusDataBox
        self.savedDataSignal = []
        self.savedDataTime = []
        self.savedDataPosition = []

        self.setWindowTitle('Focus')
        self.setAntialiasing(True)

        self.npoints = 400
        self.data = np.zeros(self.npoints)
        self.ptr = 0

        # Graph without a fixed range
        self.statistics = pg.LabelItem(justify='right')
        self.addItem(self.statistics)
        self.statistics.setText('---')
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabels(bottom=('Time', 's'),
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
        self.focusSignal = self.focusWidget.ProcessData.focusSignal

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

            if self.recButton.isChecked():
                self.analize()


class WebcamGraph(pg.GraphicsWindow):

    def __init__(self, focusWidget, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.focusWidget = focusWidget

        self.img = pg.ImageItem(border='w')
        self.view = self.addViewBox(invertY=True, invertX=False)
        self.view.setAspectLocked(True)  # square pixels
        self.view.addItem(self.img)

    def update(self):
        self.img.setImage(self.focusWidget.ProcessData.image)


class FocusCalibration(QtCore.QObject):

    def __init__(self, mainwidget, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.signalData = []
        self.positionData = []
        self.nm = Q_(1, 'nm')
        self.step = 50 * self.nm
        self.z = mainwidget.z
        self.mainwidget = mainwidget  # mainwidget será FocusLockWidget

    def start(self):

        for i in range(20):
            self.focusCalibSignal = self.mainwidget.ProcessData.focusSignal
            self.signalData.append(self.focusCalibSignal)
            self.positionData.append(self.z.position.magnitude)
            self.z.moveRelative(self.step)
            time.sleep(0.5)

        self.argmax = np.argmax(self.signalData)
        self.argmin = np.argmin(self.signalData)
        self.signalData = self.signalData[self.argmin:self.argmax]
        self.positionData = self.positionData[self.argmin:self.argmax]

        self.poly = np.polyfit(self.positionData, self.signalData, 1)
        self.calibrationResult = np.around(self.poly, 2)
        self.export()

    def export(self):

        np.savetxt('calibration', self.calibrationResult)
        cal = np.around(np.abs(self.calibrationResult[0]), 1)
        calText = '1 px --> {} nm'.format(cal)
        self.mainwidget.calibrationDisplay.setText(calText)
        poly = np.polynomial.polynomial.polyval(self.positionData,
                                                self.calibrationResult[::-1])
        self.savedCalibData = [self.positionData, self.signalData, poly]
        np.savetxt('calibrationcurves', self.savedCalibData)


if __name__ == '__main__':

    app = QtGui.QApplication([])

    with instruments.ScanZ(12) as z:
        win = FocusWidget(z)
        win.show()
        app.exec_()
