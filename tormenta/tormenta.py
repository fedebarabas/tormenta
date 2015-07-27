# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: federico
"""
from pyqtgraph.Qt import QtGui

from tormenta.control import control
import tormenta.control.instruments as instruments


def main():

    app = QtGui.QApplication([])

    with instruments.Camera('andor.ccd.CCD') as andor, \
            instruments.Laser('mpb.vfl.VFL', 'COM11') as redlaser, \
            instruments.Laser('rgblasersystems.minilasevo.MiniLasEvo', 'COM7') as bluelaser, \
            instruments.Laser('laserquantum.ventus.Ventus', 'COM13') as greenlaser, \
            instruments.DAQ() as daq, instruments.ScanZ(12) as scanZ:

        print(andor.idn)
        print(redlaser.idn)
        print(bluelaser.idn)
        print(greenlaser.idn)
        print(daq.idn)
        print('Prior Z stage')

        win = control.TormentaGUI(andor, redlaser, bluelaser, greenlaser,
                                  scanZ, daq)
        win.show()

        app.exec_()
