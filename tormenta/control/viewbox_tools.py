# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:52:49 2015

@author: Federico Barabas
"""

import numpy as np
from PyQt4 import QtCore, QtGui
import pyqtgraph as pg


# taken from https://www.mrao.cam.ac.uk/~dag/CUBEHELIX/cubehelix.py
def cubehelix(gamma=1.0, s=0.5, r=-1.5, h=1.0):
    def get_color_function(p0, p1):
        def color(x):
            xg = x ** gamma
            a = h * xg * (1 - xg) / 2
            phi = 2 * np.pi * (s / 3 + r * x)
            return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))
        return color

    array = np.empty((256, 3))
    abytes = np.arange(0, 1, 1/256.)
    array[:, 0] = get_color_function(-0.14861, 1.78277)(abytes) * 255
    array[:, 1] = get_color_function(-0.29227, -0.90649)(abytes) * 255
    array[:, 2] = get_color_function(1.97294, 0.0)(abytes) * 255
    return array


class Grid():

    def __init__(self, viewBox, shape):

        self.showed = False
        self.vb = viewBox
        self.shape = shape
        pen = QtGui.QPen(QtCore.Qt.yellow, 0.75, QtCore.Qt.DotLine)
        pen2 = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)

        self.yline1 = pg.InfiniteLine(pos=0.25*self.shape[0], pen=pen)
        self.yline2 = pg.InfiniteLine(pos=0.375*self.shape[0], pen=pen)
        self.yline3 = pg.InfiniteLine(pos=0.50*self.shape[0], pen=pen2)
        self.yline4 = pg.InfiniteLine(pos=0.625*self.shape[0], pen=pen)
        self.yline5 = pg.InfiniteLine(pos=0.75*self.shape[0], pen=pen)
        self.xline1 = pg.InfiniteLine(pos=0.25*self.shape[1], pen=pen, angle=0)
        self.xline2 = pg.InfiniteLine(pos=0.375*self.shape[1], pen=pen2,
                                      angle=0)
        self.xline3 = pg.InfiniteLine(pos=0.50*self.shape[1], pen=pen2,
                                      angle=0)
        self.xline4 = pg.InfiniteLine(pos=0.625*self.shape[1], pen=pen2,
                                      angle=0)
        self.xline5 = pg.InfiniteLine(pos=0.75*self.shape[1], pen=pen, angle=0)

    def update(self, shape):
        self.yline1.setPos(0.25*shape[0])
        self.yline2.setPos(0.375*shape[0])
        self.yline3.setPos(0.50*shape[0])
        self.yline4.setPos(0.625*shape[0])
        self.yline5.setPos(0.75*shape[0])
        self.xline1.setPos(0.25*shape[1])
        self.xline2.setPos(0.375*shape[1])
        self.xline3.setPos(0.50*shape[1])
        self.xline4.setPos(0.625*shape[1])
        self.xline5.setPos(0.75*shape[1])

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.addItem(self.xline1)
        self.vb.addItem(self.xline2)
        self.vb.addItem(self.xline3)
        self.vb.addItem(self.xline4)
        self.vb.addItem(self.xline5)
        self.vb.addItem(self.yline1)
        self.vb.addItem(self.yline2)
        self.vb.addItem(self.yline3)
        self.vb.addItem(self.yline4)
        self.vb.addItem(self.yline5)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.xline1)
        self.vb.removeItem(self.xline2)
        self.vb.removeItem(self.xline3)
        self.vb.removeItem(self.xline4)
        self.vb.removeItem(self.xline5)
        self.vb.removeItem(self.yline1)
        self.vb.removeItem(self.yline2)
        self.vb.removeItem(self.yline3)
        self.vb.removeItem(self.yline4)
        self.vb.removeItem(self.yline5)
        self.showed = False


class TwoColorGrid():

    def __init__(self, viewBox, shape=(512, 512)):

        self.showed = False
        self.vb = viewBox
        self.shape = shape

        pen = QtGui.QPen(QtCore.Qt.yellow, 1, QtCore.Qt.SolidLine)
        pen2 = QtGui.QPen(QtCore.Qt.yellow, 0.75, QtCore.Qt.DotLine)

        self.rectT = QtGui.QGraphicsRectItem(192, 192, 128, 128)
        self.rectT.setPen(pen)
        self.rectR = QtGui.QGraphicsRectItem(192, 54, 128, 128)
        self.rectR.setPen(pen)
        self.yLine = pg.InfiniteLine(pos=0.5*self.shape[0], pen=pen2)
        self.xLine = pg.InfiniteLine(pos=0.5*self.shape[1], pen=pen2, angle=0)
#        self.xLineT = pg.InfiniteLine(pos=118, pen=pen2, angle=0)
        self.xLineR = pg.InfiniteLine(pos=118, pen=pen2, angle=0)

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.addItem(self.rectT)
        self.vb.addItem(self.rectR)
        self.vb.addItem(self.yLine)
        self.vb.addItem(self.xLine)
        self.vb.addItem(self.xLineR)
#        self.vb.addItem(self.xLineT)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.rectT)
        self.vb.removeItem(self.rectR)
        self.vb.removeItem(self.yLine)
        self.vb.removeItem(self.xLine)
        self.vb.removeItem(self.xLineR)
#        self.vb.removeItem(self.xLineT)
        self.showed = False


class Crosshair():

    def __init__(self, viewBox):

        self.showed = False

        self.vLine = pg.InfiniteLine(pos=0, angle=90, movable=False)
        self.hLine = pg.InfiniteLine(pos=0, angle=0,  movable=False)
        self.vb = viewBox

    def mouseMoved(self, pos):
        if self.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())

    def mouseClicked(self):
        try:
            self.vb.scene().sigMouseMoved.disconnect(self.mouseMoved)
        except:
            pass

    def toggle(self):
        if self.showed:
            self.hide()
        else:
            self.show()

    def show(self):
        self.vb.scene().sigMouseClicked.connect(self.mouseClicked)
        self.vb.scene().sigMouseMoved.connect(self.mouseMoved)
        self.vb.addItem(self.vLine, ignoreBounds=False)
        self.vb.addItem(self.hLine, ignoreBounds=False)
        self.showed = True

    def hide(self):
        self.vb.removeItem(self.vLine)
        self.vb.removeItem(self.hLine)
        self.showed = False


class ROI(pg.ROI):

    def __init__(self, shape, vb, pos, handlePos, handleCenter, *args,
                 **kwargs):

        self.mainShape = shape

        pg.ROI.__init__(self, pos, size=(128, 128), pen='y', *args, **kwargs)
        self.addScaleHandle(handlePos, handleCenter, lockAspect=True)
        vb.addItem(self)

        self.label = pg.TextItem()
        self.label.setPos(self.pos()[0] + self.size()[0],
                          self.pos()[1] + self.size()[1])
        self.label.setText('128x128')

        self.sigRegionChanged.connect(self.updateText)

        vb.addItem(self.label)

    def updateText(self):
        self.label.setPos(self.pos()[0] + self.size()[0],
                          self.pos()[1] + self.size()[1])
        size = np.round(self.size()).astype(np.int)
        self.label.setText('{}x{}'.format(size[0], size[1]))

    def hide(self, *args, **kwargs):
        super().hide(*args, **kwargs)
        self.label.hide()


class cropROI(pg.ROI):

    def __init__(self, shape, vb, *args, **kwargs):

        self.mainShape = shape

        pg.ROI.__init__(self, pos=(shape[0], shape[1]), size=(128, 128),
                        scaleSnap=True, translateSnap=True, movable=False,
                        pen='y', *args, **kwargs)
        self.addScaleHandle((0, 1), (1, 0))
