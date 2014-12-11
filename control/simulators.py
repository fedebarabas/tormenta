# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 20:02:08 2014

@author: Federico Barabas
"""

# -*- coding: utf-8 -*-
"""
    lantz.simulators.fungen
    ~~~~~~~~~~~~~~~~~~~~~~~

    A simulated function generator.
    See specification in the Lantz documentation.

    :copyright: 2012 by The Lantz Authors
    :license: BSD, see LICENSE for more details.
"""

import logging
import numpy as np

from lantz import Driver
from lantz import Q_

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S')


class SimLaser(Driver):

    def __init__(self):
        super().__init__()

        self.mW = Q_(1, 'mW')

        self.enabled = False
        self.power_sp = 0 * self.mW
        print("Simulated laser initialized")

    @property
    def idn(self):
        return 'Simulated laser'

    @property
    def status(self):
        """Current device status
        """
        return 'Simulated laser status'

    # ENABLE LASER
    @property
    def enabled(self):
        """Method for turning on the laser
        """
        return self.enabled_state

    @enabled.setter
    def enabled(self, value):
        self.enabled_state = value

    # LASER'S CONTROL MODE AND SET POINT

    @property
    def power_sp(self):
        """To handle output power set point (mW) in APC Mode
        """
        return self.power_setpoint

    @power_sp.setter
    def power_sp(self, value):
        self.power_setpoint = value

    # LASER'S CURRENT STATUS

    @property
    def power(self):
        """To get the laser emission power (mW)
        """
        return 55555 * self.mW


class SimCamera(Driver):

    def __init__(self):
        super().__init__()

        self.degC = Q_(1, 'degC')
        self.s = Q_(1, 's')
        self.us = Q_(1, 'us')

        self.temperature_setpoint = Q_(-10, 'degC')
        self.cooler_on_state = False
        self.acq_mode = 'Run till abort'
        self.status_state = 'Camera is idle, waiting for instructions.'
        self.image_size = self.detector_shape
        self.preamp_st = 1
        self.EM_gain_st = 1
        self.ftm_state = False
        self.horiz_shift_speed_state = 1
        self.n_preamps = 1

    @property
    def idn(self):
        """Identification of the device
        """
        return 'Simulated Andor camera'

    @property
    def detector_shape(self):
        return (512, 512)

    @property
    def px_size(self):
        """ This function returns the dimension of the pixels in the detector
        in microns.
        """
        return (8, 8)

    @property
    def temperature(self):
        """ This function returns the temperature of the detector to the
        nearest degree. It also gives the status of cooling process.
        """
        return Q_(55555, 'degC')

    @property
    def temperature_setpoint(self):
        return self.temperature_sp

    @temperature_setpoint.setter
    def temperature_setpoint(self, value):
        self.temperature_sp = value

    @property
    def cooler_on(self):
        return self.cooler_on_state

    @cooler_on.setter
    def cooler_on(self, value):
        self.cooler_on_state = value

    @property
    def temperature_status(self):
        if self.cooler_on_state:
            return 'Temperature stabilized'
        else:
            return 'Temperature not stabilized'

    @property
    def acquisition_mode(self):
        """ This function will set the acquisition mode to be used on the next
        StartAcquisition.
        NOTE: In Mode 5 the system uses a “Run Till Abort” acquisition mode. In
        Mode 5 only, the camera continually acquires data until the
        AbortAcquisition function is called. By using the SetDriverEvent
        function you will be notified as each acquisition is completed.
        """
        return self.acq_mode

    @acquisition_mode.setter
    def acquisition_mode(self, mode):
        self.acq_mode = mode

    def start_acquisition(self):
        """ This function starts an acquisition. The status of the acquisition
        can be monitored via GetStatus().
        """
        self.status_state = 'Acquisition in progress.'
        self.j = 0

    def abort_acquisition(self):
        """This function aborts the current acquisition if one is active
        """
        self.status_state = 'Camera is idle, waiting for instructions.'

    @property
    def status(self):
        """ This function will return the current status of the Andor SDK
        system. This function should be called before an acquisition is started
        to ensure that it is IDLE and during an acquisition to monitor the
        process.
        """
        return self.status_state

    def set_image(self, shape, p_0):
        """ This function will set the horizontal and vertical binning to be
        used when taking a full resolution image.
        Parameters
            int hbin: number of pixels to bin horizontally.
            int vbin: number of pixels to bin vertically.
            int hstart: Start column (inclusive).
            int hend: End column (inclusive).
            int vstart: Start row (inclusive).
            int vend: End row (inclusive).
        """
        self.image_size = (shape)

    def free_int_mem(self):
        """The FreeInternalMemory function will deallocate any memory used
        internally to store the previously acquired data. Note that once this
        function has been called, data from last acquisition cannot be
        retrived.
        """
        pass

    def most_recent_image16(self, npixels):
        """ This function will update the data array with the most recently
        acquired image in any acquisition mode. The data are returned as long
        integers (32-bit signed integers). The "array" must be exactly the same
        size as the complete image.
        """
        return np.random.normal(100, 10, self.image_size).astype(np.uint16)

    def images16(self, first, last, shape, validfirst, validlast):
        arr = np.random.normal(100, 10, (last - first + 1, shape[0], shape[1]))
        return arr.astype(np.uint16)

    def set_n_kinetics(self, n):
        self.n = n

    def set_n_accum(self, i):
        pass

    def set_accum_time(self, dt):
        pass

    def set_kinetic_cycle_time(self, dt):
        pass

    @property
    def n_images_acquired(self):
        self.j += 1
        if self.j == self.n:
            self.status_state = 'Camera is idle, waiting for instructions.'
        return self.j

    @property
    def new_images_index(self):
        return (self.j, self.j)

    def shutter(self, *args):
        pass

    @property
    def preamp(self):
        return self.preamp_st

    @preamp.setter
    def preamp(self, value):
        self.preamp_st = value

    def true_preamp(self, n):
        return 10

    def n_horiz_shift_speeds(self):
        return 1

    def true_horiz_shift_speed(self, n):
        return 100 * self.us

    @property
    def horiz_shift_speed(self):
        return self.horiz_shift_speed_state

    @horiz_shift_speed.setter
    def horiz_shift_speed(self, value):
        self.horiz_shift_speed_state = value

    @property
    def max_exposure(self):
        return 12 * self.s

    @property
    def acquisition_timings(self):
        return 0.01 * self.s, 0.01 * self.s, 0.01 * self.s

    @property
    def EM_gain_range(self):
        return (0, 1000)

    @property
    def EM_gain(self):
        return self.EM_gain_st

    @EM_gain.setter
    def EM_gain(self, value):
        self.EM_gain_st = value

    @property
    def n_vert_shift_speeds(self):
        return 4

    def true_vert_shift_speed(self, n):
        return 3.3 * self.us

    @property
    def n_vert_clock_amps(self):
        return 4

    def true_vert_amp(self, n):
        return 1

    def set_vert_clock(self, n):
        pass

    def set_exposure_time(self, t):
        pass

    @property
    def frame_transfer_mode(self):
        return self.ftm_state

    @frame_transfer_mode.setter
    def frame_transfer_mode(self, state):
        self.ftm_state = state
