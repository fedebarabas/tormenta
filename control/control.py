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

    img.setImage(andor.most_recent_image16(andor.detector_shape),
                 autoLevels=False)
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


def liveview():
    """ Image live view when not recording
    """
    global andor, img, viewtimer

    if andor.status != 'Camera is idle, waiting for instructions.':
        andor.abort_acquisition()

    andor.acquisition_mode = 'Run till abort'
    andor.free_int_mem()

    andor.start_acquisition()
    time.sleep(2)
    idata = andor.most_recent_image16(andor.detector_shape)
    img.setImage(idata)
    hist.setLevels(0.8 * idata.min(), 1.2 * idata.max())

    viewtimer.start(0)


def record(n):
    """ Record an n-frames acquisition
    """
    global andor, ishape, viewtimer, img

    # Info storage
    stack = np.zeros((n, ishape[0], ishape[1]))

    # Stop the QTimer that updates the image with incoming data from the
    # 'Run till abort' acquisition mode.
    viewtimer.stop()

    # Acquisition preparation
    if andor.status != 'Camera is idle, waiting for instructions.':
        andor.abort_acquisition()

    andor.free_int_mem()
    andor.acquisition_mode = 'Kinetics'
    andor.set_n_kinetics(n)
    andor.start_acquisition()

    print('started')

    j = 0
    while j < n:
        if andor.n_images_acquired > j:
            i, j = andor.new_images_index
            stack[i - 1:j] = andor.images16(i, j, ishape, 1, n)
#            img.setImage(stack[j - 1], autoLevels=False)
            print(j)

    return stack

    print('finished')

    liveview()


if __name__ == '__main__':

    from lantz import Q_
    s = Q_(1, 's')

    with CCD() as andor:

        print(andor.idn)

        # Not-default configuration
        ishape = andor.detector_shape
        origin = (1, 1)
        andor.set_exposure_time(0.02 * s)

        # Default camera configuration
        andor.readout_mode = 'Image'
        andor.set_image(shape=ishape, p_0=origin)
        andor.trigger_mode = 'Internal'
        andor.amp_typ = 0
        andor.horiz_shift_speed = 0
        andor.vert_shift_speed = 0
#        andor.shutter(0, 5, 0, 0, 0)   # Uncomment when using for real


#        # Temperature stabilization
#        andor.temperature_setpoint = -30 * degC
#        andor.cooler_on = True
#        stable = 'Temperature has stabilized at set point.'
#        print('Temperature set point =', andor.temperature_setpoint)
#        while andor.temperature_status != stable:
#            print("Current temperature:", np.round(andor.temperature, 1))
#            time.sleep(30)
#        print('Temperature has stabilized at set point')

        #
        # GUI design
        #

        # TODO: redefine axis ticks

        # Main window
        win = QtGui.QWidget()
        win.setWindowTitle('Tormenta')

        # Widgets
        rec = QtGui.QPushButton('REC')
        fpsbox = QtGui.QLabel()

        # Image Widget
        imagewidget = pg.GraphicsLayoutWidget()
        p1 = imagewidget.addPlot()
        p1.getViewBox().setMouseMode(pg.ViewBox.RectMode)
        p1.getViewBox().setLimits(xMin=-0.5, xMax=ishape[0] - 0.5,
                                  yMin=-0.5, yMax=ishape[1] - 0.5)
        img = pg.ImageItem()
        img.translate(-0.5, -0.5)
        p1.addItem(img)
        p1.setAspectLocked(True)
        p1.setRange(xRange=(-0.5, ishape[0] - 0.5),
                    yRange=(-0.5, ishape[1] - 0.5), padding=0)
        hist = pg.HistogramLUTItem()
        hist.setImageItem(img)
        hist.autoHistogramRange = False
        imagewidget.addItem(hist)

        # Widgets' layout
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        layout.addWidget(rec, 2, 0)
        layout.addWidget(imagewidget, 1, 2, 3, 1)
        layout.addWidget(fpsbox, 0, 2)

        win.show()

        viewtimer = QtCore.QTimer()
        viewtimer.timeout.connect(updateview)

        rec.pressed.connect(lambda: record(30))

        liveview()

        # Acquisition
#        andor.start_acquisition()
#        time.sleep(2)
#        idata = andor.most_recent_image16(andor.detector_shape)
#        img.setImage(idata)
#        hist.setLevels(0.8 * idata.min(), 1.2 * idata.max())
#
#        viewtimer.start(0)

#        print('buffer size', andor.buffer_size)
#        stack = record(30, ishape)

        app.exec_()
        viewtimer.stop()
