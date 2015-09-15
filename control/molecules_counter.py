# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 09:47:28 2015

@author: Federico Barabas
"""

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.ptime as ptime

import analysis.maxima as maxima
import analysis.tools as tools


class MoleculeWidget(QtGui.QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)

        # Widgets
        self.graph = MoleculesGraph(self)
        self.enableBox = QtGui.QCheckBox('Enable')
        self.enableBox.setEnabled(False)
        self.enableBox.stateChanged.connect(self.graph.getTime)
        self.lockButton = QtGui.QPushButton('Lock (no anda)')
        self.lockButton.setCheckable(True)
        self.lockButton.clicked.connect(self.toggleLock)
        self.lockButton.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                      QtGui.QSizePolicy.Expanding)
        self.alphaLabel = QtGui.QLabel('Alpha')
        self.alphaLabel.setAlignment((QtCore.Qt.AlignRight |
                                      QtCore.Qt.AlignVCenter))
        self.alphaEdit = QtGui.QLineEdit('5')
        self.alphaEdit.setFixedWidth(40)

        # GUI layout
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.graph, 0, 0, 1, 5)
        grid.addWidget(self.enableBox, 1, 0)
        grid.addWidget(self.alphaLabel, 1, 1)
        grid.addWidget(self.alphaEdit, 1, 2)
        grid.addWidget(self.lockButton, 1, 3, 1, 2)

        grid.setColumnMinimumWidth(0, 200)
        grid.setColumnMinimumWidth(4, 170)

    @property
    def enabled(self):
        return self.enableBox.isChecked()

    def toggleLock(self):
        pass


class MoleculesGraph(pg.PlotWidget):

    def __init__(self, mainWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.main = mainWidget
        self.npoints = 200

        self.setAntialiasing(True)

        # First plot for the number of molecules
        self.plot1 = self.plotItem
        self.plot1.setLabels(bottom=('Tiempo', 's'),
                             left=('Number of single molecules'))
        self.plot1.showGrid(x=True, y=True)
        self.curve1 = self.plot1.plot(pen='y')
        self.plot1.vb.setLimits(yMin=-0.5)

        # Second plot for the number of overlaps
        self.plot2 = pg.ViewBox()
        self.ax2 = pg.AxisItem('right')
        self.plot1.layout.addItem(self.ax2, 2, 3)
        self.plot1.setLimits(yMin=-0.5)
        self.plot1.scene().addItem(self.plot2)
        self.ax2.linkToView(self.plot2)
        self.plot2.setXLink(self.plot1)
        self.ax2.setLabel('Number of overlaps')
        self.curve2 = pg.PlotCurveItem(pen='r')
        self.plot2.addItem(self.curve2)
        self.plot2.setLimits(yMin=-0.5)

        # Handle view resizing
        self.updateViews()
        self.plot1.vb.sigResized.connect(self.updateViews)

        self.fwhm = tools.get_fwhm(670, 1.42) / 120
        self.kernel = tools.kernel(self.fwhm)

    def updateViews(self):
        self.plot2.setGeometry(self.plot1.vb.sceneBoundingRect())
        self.plot2.linkedViewChanged(self.plot1.vb, self.plot2.XAxis)

    def getTime(self):
        if self.main.enabled:
            self.ptr = 0
            self.dataN = np.zeros(self.npoints, dtype=np.int)
            self.dataOverlaps = np.zeros(self.npoints, dtype=np.int)
            self.time = np.zeros(self.npoints)
            self.startTime = ptime.time()

    def update(self, image):

        peaks = maxima.Maxima(image, self.fwhm, self.kernel)
        peaks.find(float(self.main.alphaEdit.text()))
        nMaxima = len(peaks.positions)
        nOverlaps = peaks.overlaps

        print(self.ptr)

        if self.ptr < self.npoints:
            self.dataN[self.ptr] = nMaxima
            self.dataOverlaps[self.ptr] = nOverlaps
            self.time[self.ptr] = ptime.time() - self.startTime
            self.curve1.setData(self.time[1:self.ptr + 1],
                                self.dataN[1:self.ptr + 1])
            self.curve2.setData(self.time[1:self.ptr + 1],
                                self.dataOverlaps[1:self.ptr + 1])

        else:
            self.dataN[:-1] = self.dataN[1:]
            self.dataN[-1] = nMaxima
            self.dataOverlaps[:-1] = self.dataOverlaps[1:]
            self.dataOverlaps[-1] = nOverlaps
            self.time[:-1] = self.time[1:]
            self.time[-1] = ptime.time() - self.startTime

            self.curve1.setData(self.time, self.dataN)
            self.curve2.setData(self.time, self.dataOverlaps)

        self.ptr += 1
