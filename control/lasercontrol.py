# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:51:21 2014

@author: Federico Barabas
"""

import time
from PyQt4 import QtGui, QtCore
from lantz import Q_


class UpdatePowers(QtCore.QObject):

    def __init__(self, laserwidget, *args, **kwargs):

        super(QtCore.QObject, self).__init__(*args, **kwargs)
        self.widget = laserwidget

    def update(self):
        redpower = '{:~}'.format(self.widget.redlaser.power)
        bluepower = '{:~}'.format(self.widget.bluelaser.power)
        greenpower = '{:~}'.format(self.widget.greenlaser.power)
        self.widget.redControl.powerIndicator.setText(redpower)
        self.widget.blueControl.powerIndicator.setText(bluepower)
        self.widget.greenControl.powerIndicator.setText(greenpower)
        time.sleep(1)
        QtCore.QTimer.singleShot(1, self.update)


class LaserWidget(QtGui.QFrame):

    def __init__(self, lasers, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.redlaser, self.bluelaser, self.greenlaser = lasers
        self.mW = Q_(1, 'mW')

        self.redControl = LaserControl(self.redlaser,
                                       '<h3>MPB 642nm</h3>',
                                       color=(255, 11, 0), prange=(150, 1500),
                                       tickInterval=100, singleStep=10)

        self.blueControl = LaserControl(self.bluelaser,
                                        '<h3>RGB 405nm</h3>',
                                        color=(73, 0, 188),
                                        prange=(0, 100),
                                        tickInterval=10, singleStep=1)

        self.greenControl = LaserControl(self.greenlaser,
                                         '<h3>Ventus 532nm</h3>',
                                         color=(80, 255, 0),
                                         prange=(0, 1500),
                                         tickInterval=10, singleStep=1)

        self.controls = (self.redControl, self.blueControl, self.greenControl)

        self.findTirfButton = QtGui.QPushButton('Find TIRF')
        self.setEpiButton = QtGui.QPushButton('Set EPI')
        self.tirfButton = QtGui.QPushButton('TIRF')
        self.tirfButton.setCheckable(True)
        self.tirfButton.
        self.epiButton = QtGui.QPushButton('EPI')
        self.epiButton.setCheckable(True)
        self.stagePosLabel = QtGui.QLabel('0 mm')
        self.stagePosLabel.setSizePolicy(QtGui.QSizePolicy.Preferred,
                                         QtGui.QSizePolicy.Expanding)

        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.redControl, 0, 1)
        grid.addWidget(self.blueControl, 0, 0)
        grid.addWidget(self.greenControl, 0, 2)
        grid.addWidget(self.findTirfButton, 1, 0)
        grid.addWidget(self.setEpiButton, 2, 0)
        grid.addWidget(self.tirfButton, 1, 1)
        grid.addWidget(self.epiButton, 2, 1)
        grid.addWidget(self.stagePosLabel, 2, 2, 2, 1)

        # Current power update routine
        self.updatePowers = UpdatePowers(self)
        self.updateThread = QtCore.QThread()
        self.updatePowers.moveToThread(self.updateThread)
        self.updateThread.start()
        self.updateThread.started.connect(self.updatePowers.update)

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)
        self.updateThread.terminate()


class LaserControl(QtGui.QFrame):

    def __init__(self, laser, name, color, prange, tickInterval, singleStep,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Raised)
        self.laser = laser
        self.mW = Q_(1, 'mW')

        self.name = QtGui.QLabel(name)
        self.name.setTextFormat(QtCore.Qt.RichText)

        self.powerIndicator = QtGui.QLineEdit('{:~}'.format(self.laser.power))
        self.powerIndicator.setReadOnly(True)
        self.powerIndicator.setFixedWidth(100)
        self.setPointEdit = QtGui.QLineEdit(str(self.laser.power_sp.magnitude))
        self.setPointEdit.setFixedWidth(100)
        self.enableButton = QtGui.QPushButton('ON')
        self.enableButton.setFixedWidth(100)
        style = "background-color: rgb{}".format(color)
        self.enableButton.setStyleSheet(style)
        self.enableButton.setCheckable(True)
        if self.laser.enabled:
            self.enableButton.setChecked(True)

        self.maxpower = QtGui.QLabel(str(prange[1]))
        self.maxpower.setAlignment(QtCore.Qt.AlignCenter)
        self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.setMinimum(prange[0])
        self.slider.setMaximum(prange[1])
        self.slider.setTickInterval(tickInterval)
        self.slider.setSingleStep(singleStep)
        self.slider.setValue(self.laser.power.magnitude)
        self.minpower = QtGui.QLabel(str(prange[0]))
        self.minpower.setAlignment(QtCore.Qt.AlignCenter)

        grid = QtGui.QGridLayout()
        self.setLayout(grid)
        grid.addWidget(self.name, 0, 0)
        grid.addWidget(self.powerIndicator, 3, 0)
        grid.addWidget(self.setPointEdit, 4, 0)
        grid.addWidget(self.enableButton, 5, 0)
        grid.addWidget(self.maxpower, 1, 1)
        grid.addWidget(self.slider, 2, 1, 5, 1)
        grid.addWidget(self.minpower, 7, 1)

        grid.setRowMinimumHeight(2, 60)
        grid.setRowMinimumHeight(6, 60)

        # Connections
        self.enableButton.toggled.connect(self.toggleLaser)
        self.slider.valueChanged[int].connect(self.changeSlider)
        self.setPointEdit.returnPressed.connect(self.changeEdit)

    def toggleLaser(self):
        if self.enableButton.isChecked():
            self.laser.enabled = True
        else:
            self.laser.enabled = False

    def enableLaser(self):
        self.laser.enabled = True
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW

    def changeSlider(self, value):
        self.laser.power_sp = self.slider.value() * self.mW
        self.setPointEdit.setText(str(self.laser.power_sp.magnitude))

    def changeEdit(self):
        self.laser.power_sp = float(self.setPointEdit.text()) * self.mW
        self.slider.setValue(self.laser.power_sp.magnitude)

    def closeEvent(self, *args, **kwargs):
        super().closeEvent(*args, **kwargs)


if __name__ == '__main__':

    app = QtGui.QApplication([])

    from instruments import Laser

    with Laser('mpb.vfl.VFL', 'COM11') as redlaser, \
            Laser('rgblasersystems.minilasevo.MiniLasEvo', 'COM7') as bluelaser, \
            Laser('laserquantum.ventus.Ventus', 'COM13') as greenlaser:

        print(redlaser.idn, bluelaser.idn, greenlaser.idn)
        win = LaserWidget((redlaser, bluelaser, greenlaser))
        win.show()

        app.exec_()
