# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np

from lantz.drivers.andor.ccd import CCD

app = QtGui.QApplication([])

def updateview():


#def liveview():



if __name__ == '__main__':

    with CCD() as andor:



        win = QtGui.QWidget()
        win.setWindowTitle('Tormenta')

        # Widgets
        rec = QtGui.QPushButton('REC')

        imagewidget = pg.GraphicsLayoutWidget()
        view = imagewidget.addViewBox()
        view.setAspectLocked(True)
        img = pg.ImageItem(border='w')
        view.addItem(img)
        view.setRange(QtCore.QRectF(0, 0, 512, 512))

        # Layout
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        layout.addWidget(rec, 2, 0)
        layout.addWidget(imagewidget, 0, 1, 3, 1)

        win.show()

        print(andor.idn)
        app.exec_()
