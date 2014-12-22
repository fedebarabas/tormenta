# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 13:41:48 2014

@author: luciano / federico
"""

import numpy as np

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

from lantz.drivers.labjack.t7 import T7
from lantz.drivers.prior.nanoscanz import NanoScanZ

from lantz import Q_


class FocusWidget(QtGui.QFrame):

        def __init__(self, daq, scanz, *args, **kwargs):
            super(FocusWidget, self).__init__(*args, **kwargs)

            self.daq = daq
            self.z = scanz

            self.z.hostPosition = 'left'

            self.V = Q_(1, 'V')
            self.um = Q_(1, 'um')

            self.graph = FocusLockGraph(self.daq)
            self.focusTitle = QtGui.QLabel('<h2>Focus control</h2>')
            self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

            self.loadButton = QtGui.QPushButton('Load sample')
            self.loadButton.setCheckable(True)
            self.loadButton.toggled.connect(self.loadSample)

            # Layout
            grid = QtGui.QGridLayout()
            self.setLayout(grid)
            grid.addWidget(self.focusTitle, 0, 0)
            grid.addWidget(self.graph, 1, 0, 1, 4)
            grid.addWidget(self.loadButton, 2, 0)

            # Labjack configuration
            self.port = 'AIN0'
            names = [self.port + "_NEGATIVE_CH", self.port + "_RANGE",
                     "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
            # single-ended, +/-1V, 0, 0 (defaults)
            values = [self.daq.constants.GND, 0.1, 0, 0]
            self.daq.writeNames(names, values)
            scanRate = 100000
            scansPerRead = int(scanRate/10)
            portAddress = self.daq.address(self.port)[0]
            scanRate = self.daq.streamStart(scansPerRead, [portAddress],
                                            scanRate)

        def loadSample(self):
            if self.loadButton.isChecked():
                self.loadButton.setText('Loading sample')
                self.z.position = -3000 * self.um

            else:
                self.z.position = -1000 * self.um
                self.loadButton.setText('Load sample')

        def closeEvent(self, *args, **kwargs):
            self.daq.streamStop()
            self.graph.timer.stop()

            super(FocusWidget, self).closeEvent(*args, **kwargs)


class FocusLockGraph(pg.GraphicsWindow):

    def __init__(self, daq, *args, **kwargs):

        self.daq = daq

        super(FocusLockGraph, self).__init__(*args, **kwargs)
        self.setWindowTitle('Focus')
        self.setAntialiasing(True)

        self.data = np.zeros(200)
        self.ptr = 0

        # Graph with a fixed range
        self.p1 = self.addPlot()
        self.p1.setLabel('bottom', "Time")
        self.p1.setLabel('left', 'V')
        self.p1.setRange(yRange=(-0.01, 0.01))
        self.curve1 = self.p1.plot(self.data)

        # Graph without a fixed range
        self.p2 = self.addPlot()
        self.p2.setLabel('bottom', "Time")
        self.p2.setLabel('left', 'V')
        self.curve2 = self.p2.plot(self.data)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

#        self.i = 0
#        self.measurements = []
#        self.timer.timeout.connect(self.measurement)

        self.timer.start(15)

    def update(self):
        """ Gives an update of the data displayed in the graphs
        """

        self.data[:-1] = self.data[1:]  # shift data (see also: np.roll)
        newArray = self.daq.streamRead()[0]
        self.data[-1] = np.mean(newArray)
        self.ptr += 1
        self.curve1.setData(self.data)
        self.curve1.setPos(self.ptr, 0)
        self.curve2.setData(self.data)
        self.curve2.setPos(self.ptr, 0)

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

    with T7() as daq, NanoScanZ(12) as z:

        win = FocusWidget(daq, z)
        win.show()

        app.exec_()