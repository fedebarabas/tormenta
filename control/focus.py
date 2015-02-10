# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@author: luciano / federico
"""

import numpy as np
import time
import matplotlib.pyplot as plt


from PyQt4 import QtGui, QtCore
import pyqtgraph as pg


from lantz import Q_

from instruments import ScanZ, DAQ
from pi import PI
#import easygui as e


class FocusWidget(QtGui.QFrame):

    def __init__(self, DAQ, scanZ, main=None, *args, **kwargs):
        super(FocusWidget, self).__init__(*args, **kwargs)

        self.main = main
        self.DAQ = DAQ
        self.z = scanZ
        self.setPoint = 0
        self.calibrationResult = [0, 0]

        self.z.hostPosition = 'left'

        self.V = Q_(1, 'V')
        self.um = Q_(1, 'um')
        self.nm = Q_(1, 'nm')

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Thread for getting data from DAQ
        scansPerS = 5
        self.stream = daqStream(DAQ, scansPerS)
        self.streamThread = QtCore.QThread()
        self.stream.moveToThread(self.streamThread)
        self.streamThread.started.connect(self.stream.start)
        self.streamThread.start()

        self.focusCalib = focusCalibration(self)
        self.focusCalibThread = QtCore.QThread()
        self.focusCalib.moveToThread(self.focusCalibThread)
        self.focusCalibButton = QtGui.QPushButton('Calibrate')
        self.focusCalibButton.clicked.connect(self.focusCalib.start)
        self.focusCalibThread.start()
        self.calibrationDisplay = QtGui.QLineEdit('0 mV --> 0 nm')
        self.calibrationDisplay.setReadOnly(False)
        # Focus lock widgets
        self.kpEdit = QtGui.QLineEdit('26')
        self.kpEdit.textChanged.connect(self.unlockFocus)
        self.kpLabel = QtGui.QLabel('kp')
        self.kiEdit = QtGui.QLineEdit('1')
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
        self.focusPropertiesDisplay = QtGui.QLabel('  st_dev = 0    max_dev = 0')
#        self.focusPropertiesDisplay.setReadOnly(True)
#        self.focusPropertiesDisplay.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.graph = FocusLockGraph(self, main)

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.graph, 0, 0, 1, 6)
        grid.addWidget(self.focusCalibButton, 1, 0)
        grid.addWidget(self.calibrationDisplay, 2, 0)
#        grid.addWidget(self.focusAnalisisButton, 3, 0)
        grid.addWidget(self.focusPropertiesDisplay, 3, 0)
        grid.addWidget(self.kpLabel, 1, 3)
        grid.addWidget(self.kpEdit, 1, 4)
        grid.addWidget(self.kiLabel, 2, 3)
        grid.addWidget(self.kiEdit, 2, 4)
        grid.addWidget(self.lockButton, 1, 5, 2, 1)
        grid.addWidget(self.focusDataBox, 1, 2)
#        grid.addWidget(self.exportDataButton, 3, 1)

#        grid.setColumnMinimumWidth(1, 100)
#        grid.setColumnMinimumWidth(2, 40)
        grid.setColumnMinimumWidth(0, 245)
        # Labjack configuration
        self.graphTimer = QtCore.QTimer()
        self.graphTimer.timeout.connect(self.graph.update)
        self.graphTime = 1000 / scansPerS
        self.graphTimer.start(self.graphTime)

        self.lockTimer = QtCore.QTimer()
        self.lockTimer.timeout.connect(self.updatePI)
        self.locked = False

    def toggleFocus(self):
        if self.lockButton.isChecked():
            self.setPoint = self.stream.newData
            self.graph.line = self.graph.plot.addLine(y=self.setPoint, pen='r')
            self.PI = PI(self.setPoint,
                         np.float(self.kpEdit.text()),
                         np.float(self.kiEdit.text()))

            self.lockTimer.start(self.graphTime)
            self.locked = True
            self.initialZ = self.z.position

        else:
            self.unlockFocus()

    def unlockFocus(self):
        if self.locked:
            self.lockButton.setChecked(False)
            self.graph.plot.removeItem(self.graph.line)
            self.lockTimer.stop()

    def updatePI(self):
        self.distance = self.z.position - self.initialZ
        if abs(self.distance) > 10 * self.um:
            self.unlockFocus()
#            e.msgbox("Maximum error allowed exceded, "
#                     "focus control has been turned off", "Error")
        else:
            out = self.PI.update(self.stream.newData)
            self.z.moveRelative(out * self.um)

    def moveZ(self, value):
        self.z.position = value

    def closeEvent(self, *args, **kwargs):

        self.graphTimer.stop()
        if self.lockButton.isChecked():
            self.lockTimer.stop()
        self.DAQ.streamStop()
        self.streamThread.terminate()

#        self.z.position = Q_(-3000, 'um')
#
#        while self.z.position > -2800 * self.um:
#            time.sleep(1)

        super(FocusWidget, self).closeEvent(*args, **kwargs)

    def exportData(self):

        self.sizeofData = np.size(self.graph.savedDataSignal)
        self.savedData = [np.ones(self.sizeofData)*self.setPoint,
                          self.graph.savedDataSignal, self.graph.savedDataTime]
        np.savetxt('{}_focusdata'.format(self.main.filename()), self.savedData)
        self.graph.savedDataSignal = []
        self.graph.savedDataTime = []

        self.plot = plt.plot(self.graph.savedDataTime,
                             self.graph.savedDataSignal, 'b-',
                             self.graph.savedDataTime,
                             np.ones(self.sizeofData)*self.setPoint, 'r-')

#        self.graph.savedDataPosition = []


    def analizeFocus(self):
#        self.rawData = np.loadtxt('focus_data')
#        self.analisisSetPoint = self.rawData[0]
#        self.plot = plt.plot(self.rawData[2], self.graph.savedDataSignal, 'b-',
#                             self.rawData[2], self.analisisSetPoint, 'r-')

        self.mean = np.around(np.mean(self.graph.savedDataSignal), 3)
        self.std_dev = np.around(np.std(self.graph.savedDataSignal), 5)
        self.max_dev = np.around(np.max(np.abs(np.array(self.graph.savedDataSignal)
                                     - self.setPoint)), 5)

        self.focusPropertiesDisplay.setText('  st_dev = {}    max_dev = {}'.format(self.std_dev, self.max_dev))


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, mainwidget, main=None, *args, **kwargs):

        self.main = main
        self.stream = mainwidget.stream
        self.analize = mainwidget.analizeFocus
        self.focusDataBox = mainwidget.focusDataBox
        self.savedDataSignal = []
        self.savedDataTime = []
        self.savedDataPosition = []

        super(FocusLockGraph, self).__init__(*args, **kwargs)
        self.setWindowTitle('Focus')
        self.setAntialiasing(True)

        self.data = np.zeros(200)
        self.ptr = 0

        # Graph without a fixed range
        self.plot = self.addPlot()
        self.plot.setLabels(bottom=('Tiempo', 's'),
                            left=('Señal de foco', 'V'))
        self.plot.showGrid(x=True, y=True)
        self.focusCurve = self.plot.plot(pen='y')
        self.scansPerS = self.stream.scansPerS
        self.xData = np.arange(0, 200/self.scansPerS, 1/self.scansPerS)

    def update(self):
        """ Gives an update of the data displayed in the graphs
        """

        if self.ptr < 200:
            self.data[self.ptr] = self.stream.newData
            self.focusCurve.setData(self.xData[1:self.ptr + 1],
                                    self.data[1:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.stream.newData

            self.focusCurve.setData(self.xData, self.data)
            self.focusCurve.setPos(self.ptr/self.scansPerS - 50, 0)

        self.ptr += 1

        if self.main is not None:

            self.recButton = self.main.recButton

            if self.recButton.isChecked():
                self.savedDataSignal.append(self.stream.newData)
                self.savedDataTime.append(self.ptr/self.scansPerS)
#               self.savedDataPosition.append(self.DAQ.position)

            if self.recButton.isChecked():
                self.analize()


class focusCalibration(QtCore.QObject):

    def __init__(self, mainwidget, *args, **kwargs):

        super(focusCalibration, self).__init__(*args, **kwargs)
        self.signalData = []
        self.positionData = []
        self.nm = Q_(1, 'nm')
        self.step = 40*self.nm
        self.stream = mainwidget.stream
        self.z = mainwidget.z
        self.mainwidget = mainwidget

    def start(self):

        for i in range(50):

            self.signalData.append(self.stream.newData)
            self.positionData.append(self.z.position.magnitude)
            self.z.moveRelative(self.step)
            time.sleep(0.5)

        self.calibrationResult = np.around(np.polyfit(np.array(self.signalData),
                                           np.array(self.positionData), 1), 2)

        self.mainwidget.calibrationDisplay.setText('0,1 mV --> {} nm'.format(np.abs(self.calibrationResult[0])*0.1))


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

    with DAQ() as DAQ, ScanZ(12) as z:

        win = FocusWidget(DAQ, z)
        win.show()

        app.exec_()
