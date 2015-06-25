# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:47:28 2015

@author: Federico Barabas
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import pyqtgraph.ptime as ptime


class MoleculeWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Widgets
        self.graph = MoleculesGraph(self)
        self.lockButton = QtGui.QPushButton('Lock (no anda)')
        self.lockButton.setCheckable(True)
        self.lockButton.clicked.connect(self.toggleLock)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.enableBox = QtGui.QCheckBox('Enable')
        self.enableBox.setEnabled(False)
        self.enableBox.stateChanged.connect(self.graph.getTime)

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.graph, 0, 0, 1, 3)
        grid.addWidget(self.enableBox, 1, 0)
        grid.addWidget(self.lockButton, 1, 1)

    @property
    def enabled(self):
        return self.enableBox.isChecked()

    def toggleLock(self):
        pass


class MoleculesGraph(pg.GraphicsWindow):

    def __init__(self, mainWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = mainWidget

        self.setAntialiasing(True)
        self.plot = self.addPlot(row=1, col=0)
        self.plot.setLabels(bottom=('Tiempo', 's'),
                            left=('Number of single molecules'))
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen='y')

        self.ptr = 0
        self.npoints = 200
        self.data = np.zeros(self.npoints, dtype=np.int)
        self.time = np.zeros(self.npoints)
        self.startTime = ptime.time()

    def getTime(self):
        if self.main.enabled:
            self.startTime = ptime.time()

    def update(self, image):

        self.nMolecules = np.int(np.random.rand() * 10)

        if self.ptr < self.npoints:
            self.data[self.ptr] = self.nMolecules
            self.time[self.ptr] = ptime.time() - self.startTime
            self.curve.setData(self.time[1:self.ptr + 1],
                               self.data[1:self.ptr + 1])

        else:
            self.data[:-1] = self.data[1:]
            self.data[-1] = self.nMolecules
            self.time[:-1] = self.time[1:]
            self.time[-1] = ptime.time() - self.startTime

            self.curve.setData(self.time, self.data)

        self.ptr += 1
