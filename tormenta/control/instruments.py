# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:25:27 2014

@author: Federico Barabas
"""

import numpy as np
import importlib

from PyQt4 import QtCore

import pygame
import pygame.camera

from lantz.drivers.legacy.andor.ccd import CCD
from lantz.drivers.legacy.labjack.t7 import T7
from lantz import Q_

import tormenta.control.mockers as mockers


class Webcam(object):

    def __new__(cls, *args):
        try:
            pygame.camera.init()
            webcam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
            webcam.start()
            return webcam

        except:
            return mockers.MockWebcam()


class Laser(object):

    def __new__(cls, iName, *args):

        try:
            pName, driverName = iName.rsplit('.', 1)
            package = importlib.import_module('lantz.drivers.legacy.' + pName)
            driver = getattr(package, driverName)
            laser = driver(*args)
            laser.initialize()
            return driver(*args)

        except:
            return mockers.MockLaser()


class DAQ(object):
    def __new__(cls, *args):

        try:
            from labjack import ljm
            handle = ljm.openS("ANY", "ANY", "ANY")
            ljm.close(handle)
            return STORMDAQ(*args)

        except:
            return mockers.MockDAQ()


class STORMDAQ(T7):
    """ Subclass of the Labjack lantz driver. """
    def __init__(self, *args):

        super().__init__(*args)
        super().initialize(*args)

        # Clock configuration for the flipper
        self.writeName("DIO_EF_CLOCK0_ENABLE", 0)
        self.writeName("DIO_EF_CLOCK0_DIVISOR", 1)
        self.writeName("DIO_EF_CLOCK0_ROLL_VALUE", 1600000)
        self.writeName("DIO_EF_CLOCK0_ENABLE", 1)
        self.writeName("DIO2_EF_ENABLE", 0)
        self.writeName("DIO2_EF_INDEX", 0)
        self.writeName("DIO2_EF_OPTIONS", 0)
        self.flipperState = True
        self.flipper = self.flipperState
        self.writeName("DIO2_EF_ENABLE", 1)

    @property
    def flipper(self):
        """ Flipper True means the ND filter is in the light path."""
        return self.flipperState

    @flipper.setter
    def flipper(self, value):
        if value:
            self.writeName("DIO2_EF_CONFIG_A", 150000)
        else:
            self.writeName("DIO2_EF_CONFIG_A", 70000)

        self.flipperState = value

    def toggleFlipper(self):
        self.flipper = not(self.flipper)


class daqStream(QtCore.QObject):
    """This stream gets an analog input at a high frame rate and returns the
    mean of the set of taken data."""
    def __init__(self, DAQ, scansPerS, port, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.DAQ = DAQ
        self.scansPerS = scansPerS
        self.port = 'AIN{}'.format(port)
        names = [self.port + "_NEGATIVE_CH", self.port + "_RANGE",
                 "STREAM_SETTLING_US", "STREAM_RESOLUTION_INDEX"]
        # single-ended, +/-1V, 0, 0 (defaults)
        # Voltage Ranges: ±10V, ±1V, ±0.1V, and ±0.01V
        values = [self.DAQ.constants.GND, 10.0, 0, 0]
        self.DAQ.writeNames(names, values)
        self.newData = 0

    def start(self):
        scanRate = 1000
        scansPerRead = int(scanRate/self.scansPerS)
        scanRate = self.DAQ.streamStart(scansPerRead,
                                        [self.DAQ.address(self.port)[0]],
                                        scanRate)

    def startTimer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(0)

    def update(self):
        self.newData = np.mean(self.DAQ.streamRead()[0])

    def getNewData(self):
        return np.mean(self.DAQ.streamRead()[0])

    def stop(self):
        try:
            self.timer.stop()
        except AttributeError:
            pass
        self.DAQ.streamStop()


class ScanZ(object):
    def __new__(cls, *args):

        try:
            from lantz.drivers.prior.nanoscanz import NanoScanZ
            scan = NanoScanZ(*args)
            scan.initialize()
            return scan

        except:
            return mockers.MockScanZ()


class Camera(object):
    """ Buffer class for testing whether the camera is connected. If it's not,
    it returns a dummy class for program testing. """

    def __new__(cls, iName, *args):

        try:
            pName, driverName = iName.rsplit('.', 1)
            package = importlib.import_module('lantz.drivers.legacy.' + pName)
            driver = getattr(package, driverName)
            camera = driver(*args)
            camera.lib.Initialize()

        except OSError as err:
            print("OS error: {0}".format(err))
            return mockers.MockCamera()

        else:
            return STORMCamera(*args)


class STORMCamera(CCD):
    """ Subclass of the Andor's lantz driver. It adapts to our needs the whole
    functionality of the camera. """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        super().initialize(*args, **kwargs)

        self.s = Q_(1, 's')
        self.mock = False

        # Default imaging parameters
        self.readout_mode = 'Image'
        self.trigger_mode = 'Internal'
        self.EM_advanced_enabled = False
        self.EM_gain_mode = 'RealGain'
        self.amp_typ = 0
        self.set_accum_time(0 * self.s)          # Minimum accumulation and
        self.set_kinetic_cycle_time(0 * self.s)  # kinetic times

        # Lists needed for the ParameterTree
        self.PreAmps = np.around([self.true_preamp(n)
                                  for n in np.arange(self.n_preamps)],
                                 decimals=1)[::-1]
        self.HRRates = [self.true_horiz_shift_speed(n)
                        for n in np.arange(self.n_horiz_shift_speeds())][::-1]
        self.vertSpeeds = [np.round(self.true_vert_shift_speed(n), 1)
                           for n in np.arange(self.n_vert_shift_speeds)]
        self.vertAmps = ['+' + str(self.true_vert_amp(n))
                         for n in np.arange(self.n_vert_clock_amps)]
        self.vertAmps[0] = 'Normal'
