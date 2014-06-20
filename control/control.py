# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 18:19:24 2014

@author: Federico Barabas
"""

import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.ptime as ptime

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


if __name__ == '__main__':

    import time
    from lantz import Q_
    s = Q_(1, 's')

    with CCD() as andor:

        win = QtGui.QWidget()
        win.setWindowTitle('Tormenta')

        # Widgets
        rec = QtGui.QPushButton('REC')

#        imagewidget = pg.GraphicsLayoutWidget()
        img = pg.ImageView()
#        view = imagewidget.addViewBox()
#        view.setAspectLocked(True)
#        img = pg.ImageItem(border='w')
#        view.addItem(img)
#        view.setRange(QtCore.QRectF(0, 0, 512, 512))

        fpsbox = QtGui.QLabel()

        # Widget's layout
        layout = QtGui.QGridLayout()
        win.setLayout(layout)
        layout.addWidget(rec, 2, 0)
        layout.addWidget(img, 1, 2, 3, 1)
        layout.addWidget(fpsbox, 0, 2)

        win.show()

        print(andor.idn)

        # Camera configuration
        andor.readout_mode = 'Image'
        andor.set_image()
#        andor.acquisition_mode = 'Single Scan'
        andor.acquisition_mode = 'Run till abort'
        andor.set_exposure_time(0.03 * s)
        andor.trigger_mode = 'Internal'
        andor.amp_typ = 0
        andor.horiz_shift_speed = 0
        andor.vert_shift_speed = 0

        # Acquisition
        andor.start_acquisition()
        time.sleep(2)
        viewtimer = QtCore.QTimer()
        viewtimer.timeout.connect(updateview)
        viewtimer.start(0)

        app.exec_()
        viewtimer.stop()
