# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@author: luciano / federico
"""

import numpy as np
import time

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

from lantz.drivers.labjack.t7 import T7
from lantz.drivers.prior.nanoscanz import NanoScanZ

from lantz import Q_

# TODO: LOCK button, fix x-axis to time in seconds


class FocusWidget(QtGui.QFrame):

    def __init__(self, DAQ, scanZ, *args, **kwargs):
        super(FocusWidget, self).__init__(*args, **kwargs)

        self.DAQ = DAQ
        self.z = scanZ

        self.z.hostPosition = 'left'

        self.V = Q_(1, 'V')
        self.um = Q_(1, 'um')

        scansPerS = 5
        self.stream = daqStream(DAQ, scansPerS)
        self.streamThread = QtCore.QThread()
        self.stream.moveToThread(self.streamThread)
        self.streamThread.started.connect(self.stream.start)
        self.streamThread.start()

        self.graph = FocusLockGraph(self.stream)
        self.focusTitle = QtGui.QLabel('<h2>Focus control</h2>')
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        self.loadButton = QtGui.QPushButton('Bajar objetivo')
        self.loadButton.pressed.connect(lambda: self.moveZ(-3000 * self.um))
        self.liftButton = QtGui.QPushButton('Subir objetivo')
        self.liftButton.pressed.connect(lambda: self.moveZ(-1000 * self.um))

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.focusTitle, 0, 0)
        grid.addWidget(self.graph, 1, 0, 1, 4)
        grid.addWidget(self.loadButton, 3, 0)
        grid.addWidget(self.liftButton, 2, 0)

        # Labjack configuration
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.graph.update)
        self.timer.start(1000 / scansPerS)

#        self.i = 0
#        self.measurements = []
#        self.timer.timeout.connect(self.measurement)

    def moveZ(self, value):
        self.z.position = value

    def closeEvent(self, *args, **kwargs):
        self.z.position = -3000 * self.um

        while self.z.position > -2800 * self.um:
            time.sleep(1)

        self.DAQ.streamStop()
        self.streamThread.terminate()
        self.timer.stop()

        super(FocusWidget, self).closeEvent(*args, **kwargs)


class daqStream(QtCore.QObject):

    def __init__(self, DAQ, scansPerS, *args, **kwargs):

        super(daqStream, self).__init__(*args, **kwargs)

        self.DAQ = DAQ
        self.scansPerS = scansPerS
        self.port = 'AIN0'
        names = [self.port + "_NEGATIVE_CH", self.port + "_RANGE",
                 "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
        # single-ended, +/-1V, 0, 0 (defaults)
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

    def update(self):
        self.newData = np.mean(self.DAQ.streamRead()[0])


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, stream, *args, **kwargs):

        self.stream = stream

        super(FocusLockGraph, self).__init__(*args, **kwargs)
        self.setWindowTitle('Focus')
        self.setAntialiasing(True)

        self.data = np.zeros(200)
        self.ptr = 0

        # Graph with a fixed range
#        self.p1 = self.addPlot()
#        self.p1.setLabel('bottom', "Time")
#        self.p1.setLabel('left', 'V')
#        self.p1.setRange(yRange=(-0.01, 0.01))
#        self.curve1 = self.p1.plot()

        # Graph without a fixed range
        self.p2 = self.addPlot()
        self.p2.setLabel('bottom', "Time")
        self.p2.setLabel('left', 'V')
        self.curve2 = self.p2.plot()

    def update(self):
        """ Gives an update of the data displayed in the graphs
        """
        if self.ptr < 200:
            self.data[self.ptr] = self.stream.newData
#            self.curve1.setData(self.data[1:self.ptr])
            self.curve2.setData(self.data[1:self.ptr])

        else:
            self.data[:-1] = self.data[1:]  # shift data (see also: np.roll)
            self.data[-1] = self.stream.newData
#            self.curve1.setData(self.data)
            self.curve2.setData(self.data)
#            self.curve1.setPos(self.ptr - 200, 0)
            self.curve2.setPos(self.ptr - 200, 0)

        self.ptr += 1

#    def measurement(self):
#
#        if self.i % 100 == 0:
#            # el problema es que se pasa del rango porque i crece
#            # indefinidamente y data sólo tiene 200 elementos
#            self.measurements.append(self.data[self.i])
#
#        self.i += 1


if __name__ == '__main__':

    app = QtGui.QApplication([])

    with T7() as DAQ, NanoScanZ(12) as z:

        win = FocusWidget(DAQ, z)
        win.show()

        app.exec_()