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
from lantz.simulators.instrument import SimError, InstrumentHandler, main_tcp, main_serial

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S')

mW = Q_(1, 'mW')


class SimLaser(Driver):

    def __init__(self):
        super().__init__()

        self.enabled = False
        self.power_sp = 0 * mW
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
        if self.power_sp > 0 * mW:
            return np.random.normal(self.power_sp, self.power_sp / 10) * mW
        else:
            return 0 * mW

degC = Q_(1, 'degC')


class SimCamera(Driver):

    def __init__(self):
        super().__init__()

        self.temperature_setpoint = -10 * degC
        self.cooler_on_state = False
        self.acq_mode = 'Run till abort'
        self.status_state = 'Camera is idle, waiting for instructions.'
        self.image_size = self.detector_size

        print("Simulated camera initialized")

    @property
    def idn(self):
        """Identification of the device
        """
        return 'Simulated Andor camera'

    @property
    def detector_size(self):
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
        if self.cooler_on_state:
            return np.random.normal(self.temperature_setpoint,
                                    self.temperature_setpoint / 19) * degC
        else:
            return 21 * degC

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
        return np.random.normal(100, 10, self.image_size)

def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Function Generator Simulator')
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser('serial')
    subparser.add_argument('-p', '--port', type=str, default='1',
                            help='Serial port')
    subparser.set_defaults(func=main_serial)

    subparser = subparsers.add_parser('tcp')
    subparser.add_argument('-H', '--host', type=str, default='localhost',
                           help='TCP hostname')
    subparser.add_argument('-p', '--port', type=int, default=5678,
                            help='TCP port')
    subparser.set_defaults(func=main_tcp)

    instrument = SimLaser()
    args = parser.parse_args(args)
    server = args.func(instrument, args)

    logging.info('interrupt the program with Ctrl-C')
    print(instrument.idn)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info('Ending')
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()