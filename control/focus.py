# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@author: luciano / federico
"""

import numpy as np

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

from lantz import Q_

from instruments import ScanZ, DAQ
from pi import PI


class FocusWidget(QtGui.QFrame):

    def __init__(self, DAQ, scanZ, *args, **kwargs):
        super(FocusWidget, self).__init__(*args, **kwargs)

        self.DAQ = DAQ
        self.z = scanZ

        self.z.hostPosition = 'left'

        self.V = Q_(1, 'V')
        self.um = Q_(1, 'um')

        self.focusTitle = QtGui.QLabel('<h2>Focus control</h2>')
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Thread for getting data from DAQ
        scansPerS = 5
        self.stream = daqStream(DAQ, scansPerS)
        self.streamThread = QtCore.QThread()
        self.stream.moveToThread(self.streamThread)
        self.streamThread.started.connect(self.stream.start)
        self.streamThread.start()

        self.graph = FocusLockGraph(self)

        # Z moving widgets
        self.loadButton = QtGui.QPushButton('Bajar objetivo')
        self.loadButton.pressed.connect(lambda: self.moveZ(-4000 * self.um))
        self.liftButton = QtGui.QPushButton('Subir objetivo')
        self.liftButton.pressed.connect(lambda: self.moveZ(-3000 * self.um))

        # Focus lock widgets
        self.kpEdit = QtGui.QLineEdit('25')
        self.kpEdit.textChanged.connect(self.unlockFocus)
        self.kpLabel = QtGui.QLabel('kp')
        self.kiEdit = QtGui.QLineEdit('0.8')
        self.kiEdit.textChanged.connect(self.unlockFocus)
        self.kiLabel = QtGui.QLabel('ki')
        self.lockButton = QtGui.QPushButton('Lock')
        self.lockButton.setCheckable(True)
        self.lockButton.clicked.connect(self.toggleFocus)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.focusTitle, 0, 0)
        grid.addWidget(self.graph, 1, 0, 1, 5)
        grid.addWidget(self.liftButton, 2, 0)
        grid.addWidget(self.loadButton, 3, 0)
        grid.addWidget(self.kpLabel, 2, 2)
        grid.addWidget(self.kpEdit, 2, 3)
        grid.addWidget(self.kiLabel, 3, 2)
        grid.addWidget(self.kiEdit, 3, 3)
        grid.addWidget(self.lockButton, 2, 4, 2, 1)
        grid.setColumnMinimumWidth(1, 500)

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

        else:
            self.unlockFocus()

    def unlockFocus(self):
        # np.savetxt('medicion_foco', self.stream.measure)
        if self.locked:
            self.lockButton.setChecked(False)
            self.graph.plot.removeItem(self.graph.line)
            self.lockTimer.stop()

    def updatePI(self):
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


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, main, *args, **kwargs):

        self.stream = main.stream

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
#        self.measure = []

    def update(self):
        self.newData = np.mean(self.DAQ.streamRead()[0])
#        self.measure.append(self.newData)

if __name__ == '__main__':

    app = QtGui.QApplication([])

    with DAQ() as DAQ, ScanZ(12) as z:

        win = FocusWidget(DAQ, z)
        win.show()

        app.exec_()
