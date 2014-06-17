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
#import pyqtgraph.ptime as ptime
#data = np.random.normal(size=(15, 600, 600), loc=1024, scale=64).astype(np.uint16)
#i = 0
#updateTime = ptime.time()
#fps = 0


def updateview():
    global img, andor

    img.setImage(andor.most_recent_image(andor.detector_shape))
    print(andor.most_recent_image(andor.detector_shape))
    QtCore.QTimer.singleShot(100, updateview)

#    global updateTime, fps, i

#    img.setImage(data[i])
#    i = (i+1) % data.shape[0]
#
#    QtCore.QTimer.singleShot(1, updateview)
#    now = ptime.time()
#    fps2 = 1.0 / (now-updateTime)
#    updateTime = now
#    fps = fps * 0.9 + fps2 * 0.1


if __name__ == '__main__':

    from lantz import Q_
    s = Q_(1, 's')
    import time

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
        andor.readout_mode = 'Image'
        andor.set_image()
#        andor.acquisition_mode = 'Single Scan'
        andor.acquisition_mode = 'Run till abort'
        andor.set_exposure_time(0.03 * s)
        andor.trigger_mode = 'Internal'
        andor.amp_typ = 0
        andor.horiz_shift_speed = 0
        andor.vert_shift_speed = 0

        andor.start_acquisition()

        updateview()
        time.sleep(60)
        andor.abort_acquisition()


        app.exec_()
