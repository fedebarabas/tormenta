# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:47:36 2013

@author: federico
"""

from __future__ import division, with_statement, print_function

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
import numpy as np
import h5py as hdf

from pyqtgraph.dockarea import *


class Stack(object):
    """Measurement stored in a hdf5 file"""

    def __init__(self, filename, imagename='frames'):

        hdffile = hdf.File(filename, 'r')

        self.image = hdffile[imagename]
        self.size = hdffile[imagename].shape[1:3]
        self.nframes = hdffile[imagename].shape[0]
        self.frame = 0


class Crosshair(pg.GraphicsObject):
    def paint(self, p, *args):
        p.setPen(pg.mkPen('g'))
        p.drawLine(-2, 0, 2, 0)
        p.drawLine(0, -2, 0, 2)

    def boundingRect(self):
        return QtCore.QRectF(-2, -2, 4, 4)

    def mouseDragEvent(self, ev):
        ev.accept()
        if ev.isStart():
            self.startPos = self.pos()
        self.setPos(self.startPos + ev.pos() - ev.buttonDownPos())


class TormentaGui(QtGui.QMainWindow):

    def __init__(self, *args, **kwargs):

        super(QtGui.QMainWindow, self).__init__(*args, **kwargs)

        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(QtGui.qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        area = DockArea()
        self.setCentralWidget(area)
        self.resize(1200, 700)
        self.setWindowTitle('Tormenta')

        ## Create docks, place them into the window one at a time.
        ## Note that size arguments are only a suggestion; docks will still
        ## have to fill the entire dock area and obey the limits of their
        ## internal widgets.
        d2 = Dock("Dock2 - Console", size=(500, 300))
        d3 = Dock("Dock3", size=(500, 400))
        d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200))
        d5 = Dock("Dock5 - Image", size=(700, 500))
        d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200))
        area.addDock(d2, 'right')
        area.addDock(d3, 'top', d2)
        area.addDock(d4, 'above', d3)
        area.addDock(d5, 'left')
        area.addDock(d6, 'above', d3)

        ## Add widgets into each dock
        w2 = pg.console.ConsoleWidget()
        d2.addWidget(w2)

        ## Hide title bar on dock 3
        w3 = pg.PlotWidget(title="Plot inside dock with no title bar")
        w3.plot(np.random.normal(size=100))
        d3.addWidget(w3)

        w4 = pg.PlotWidget(title="Dock 4 plot")
        w4.plot(np.random.normal(size=100))
        d4.addWidget(w4)

        stack = Stack('test.hdf5')

        w5 = pg.ImageView(view=pg.PlotItem())
        w5.setImage(stack.image[0])
        frameText = pg.TextItem(text='Frame 0')
        w5.getView().addItem(frameText)
        d5.addWidget(w5)

        def gotoframe(n):
            if n > 0 and n < stack.nframes - 1:
                stack.frame = n

            elif n <= 0:
                stack.frame = 0

            elif n >= stack.nframes - 1:
                stack.frame = stack.nframes - 1

            w5.setImage(stack.image[stack.frame])
            frameText.setText('Frame {}'.format(stack.frame))

        c = Crosshair()
        w5.getView().addItem(c)

        # Frame changing actions
        next_frame = QtGui.QShortcut(self)
        next_frame.setKey('Right')
        next_frame.activated.connect(lambda: gotoframe(stack.frame + 1))
        prev_frame = QtGui.QShortcut(self)
        prev_frame.setKey('Left')
        prev_frame.activated.connect(lambda: gotoframe(stack.frame - 1))
        jump250 = QtGui.QShortcut(self)
        jump250.setKey('Ctrl+Right')
        jump250.activated.connect(lambda: gotoframe(stack.frame + 250))
        jumpm250 = QtGui.QShortcut(self)
        jumpm250.setKey('Ctrl+Left')
        jumpm250.activated.connect(lambda: gotoframe(stack.frame - 250))

        w6 = pg.PlotWidget(title="Dock 6 plot")
        w6.plot(np.random.normal(size=100))
        d6.addWidget(w6)


if __name__ == '__main__':

    app = QtGui.QApplication([])

    win = TormentaGui()
    win.show()

    exit(app.exec_())
