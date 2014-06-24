# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
import time

from lantz.drivers.andor.ccd import CCD

app = QtGui.QApplication([])

lastTime = ptime.time()
fps = None


def updateview():
    global fpsbox, img, andor, lastTime, fps

    img.setImage(andor.most_recent_image(andor.detector_shape),
                 autoHistogramRange=False)
#    QtCore.QTimer.singleShot(1, updateview)
    now = ptime.time()
    dt = now - lastTime
    lastTime = now
    if fps is None:
        fps = 1.0/dt
    else:
        s = np.clip(dt*3., 0, 1)
        fps = fps * (1-s) + (1.0/dt) * s
    fpsbox.setText('%0.2f fps' % fps)


def record(n, shape):
    global andor

    # Acquisition preparation
    if andor.status != 'Camera is idle, waiting for instructions.':
        andor.abort_acquisition()

    stack = np.zeros((n, shape[0], shape[1]))

    andor.free_int_mem()
    andor.start_acquisition()

    j = 0
    while j < n:
        if andor.n_images_acquired > j:
            i, j = andor.new_images_index
            stack[i - 1:j] = andor.images(i, j, shape, 1, n)

    return stack

if __name__ == '__main__':

    from lantz import Q_
    s = Q_(1, 's')

    with CCD() as andor:

        print(andor.idn)

        # Camera configuration
        andor.readout_mode = 'Image'
        ishape = andor.detector_shape
        andor.set_image()
#        andor.acquisition_mode = 'Run till abort'
        andor.acquisition_mode = 'Kinetics'
        andor.set_exposure_time(0.2 * s)
        andor.set_n_kinetics(30)
        andor.trigger_mode = 'Internal'
        andor.amp_typ = 0
        andor.horiz_shift_speed = 0
        andor.vert_shift_speed = 0
#        andor.shutter(0, 0, 0, 0, 0)

        andor.free_int_mem()

        win = QtGui.QWidget()
        win.setWindowTitle('Tormenta')

        # Widgets
        rec = QtGui.QPushButton('REC')

#        img = pg.ImageView()
        imagewidget = pg.GraphicsLayoutWidget()
        view = imagewidget.addViewBox()
        view.setAspectLocked(True)
        img = pg.ImageItem(border='w')
        view.addItem(img)
        view.setRange(QtCore.QRectF(0, 0, ishape[0], ishape[1]))

        fpsbox = QtGui.QLabel()

        # Widget's layout
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        layout.addWidget(rec, 2, 0)
        layout.addWidget(imagewidget, 1, 2, 3, 1)
        layout.addWidget(fpsbox, 0, 2)

        win.show()

#        # Temperature stabilization
#        andor.temperature_setpoint = -30 * degC
#        andor.cooler_on = True
#        stable = 'Temperature has stabilized at set point.'
#        print('Temperature set point =', andor.temperature_setpoint)
#        while andor.temperature_status != stable:
#            print("Current temperature:", np.round(andor.temperature, 1))
#            time.sleep(30)
#        print('Temperature has stabilized at set point')

        # Acquisition
#        andor.shutter(0, 5, 0, 0, 0)
#        andor.start_acquisition()
#        viewtimer = QtCore.QTimer()
#        viewtimer.timeout.connect(updateview)
#        viewtimer.start(0)

        print('buffer size', andor.buffer_size)
        stack = record(30, ishape)

        app.exec_()
#        viewtimer.stop()
