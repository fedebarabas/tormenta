# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 13:25:27 2014

@author: federico
"""

import numpy as np
import importlib

from lantz.drivers.andor.ccd import CCD
from lantz import Q_

import pygame
import pygame.camera

from mockers import MockCamera, MockScanZ, MockLaser, MockDAQ, MockWebcam


class Webcam(object):

    def __new__(cls, *args):
        try:
            pygame.camera.init()
            webcam = pygame.camera.Camera(pygame.camera.list_cameras()[0])
            webcam.start()
            return webcam

        except:
            return MockWebcam()


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
            return MockLaser()


class DAQ(object):
    def __new__(cls, *args):

        try:
            from labjack import ljm
            handle = ljm.openS("ANY", "ANY", "ANY")
            ljm.close(handle)

            from lantz.drivers.labjack.t7 import T7
            return T7(*args)

        except:
            return MockDAQ()


class ScanZ(object):
    def __new__(cls, *args):

        try:
            from lantz.drivers.prior.nanoscanz import NanoScanZ
            scan = NanoScanZ(*args)
            scan.initialize()
            return scan

        except:
            return MockScanZ()


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
            return MockCamera()

        else:
            return STORMCamera(*args)


class STORMCamera(CCD):
    """ Subclass of the Andor's lantz driver. It adapts to our needs the whole
    functionality of the camera. """

    def __init__(self, *args, **kwargs):

        super(STORMCamera, self).__init__(*args, **kwargs)
        super(STORMCamera, self).initialize(*args, **kwargs)

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
