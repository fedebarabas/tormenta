# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:25:27 2014

@author: federico
"""

import numpy as np
import importlib
import sys

from lantz.drivers.andor.ccd import CCD
from lantz.drivers.labjack.t7 import T7
from lantz import Q_

import pygame
import pygame.camera

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
            package = importlib.import_module('lantz.drivers.' + pName)
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
            print(sys.exc_info()[0])
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
        """ Flipper ON means the ND filter is in the light path."""
        return self.flipperState

    @flipper.setter
    def flipper(self, value):
        if value:
            self.writeName("DIO2_EF_CONFIG_A", 150000)
        else:
            self.writeName("DIO2_EF_CONFIG_A", 72000)

        self.flipperState = value


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
            package = importlib.import_module('lantz.drivers.' + pName)
            driver = getattr(package, driverName)
            camera = driver(*args)
            camera.lib.Initialize()

        except:
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
