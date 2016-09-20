# -*- coding: utf-8 -*-
"""
Created on Thu May 21 13:19:31 2015

@author: Federico Barabas
"""
from pyqtgraph.Qt import QtGui


def main():

    from tormenta.control import control
    import tormenta.control.instruments as instruments

    app = QtGui.QApplication([])

    with instruments.Camera('andor.ccd.CCD') as andor, \
            instruments.Laser('mpb.vfl.VFL', 'COM3') as redlaser, \
            instruments.Laser('rgblasersystems.minilasevo.MiniLasEvo', 'COM7') as bluelaser, \
            instruments.Laser('laserquantum.ventus.Ventus', 'COM13') as greenlaser, \
            instruments.DAQ() as daq, instruments.ScanZ(4) as scanZ:

        aptMotor = instruments.Motor()

        print(andor.idn)
        print(redlaser.idn)
        print(bluelaser.idn)
        print(greenlaser.idn)
        print(daq.idn)
        print(aptMotor.getHardwareInformation())

        win = control.TormentaGUI(andor, redlaser, bluelaser, greenlaser,
                                  scanZ, daq, aptMotor)
        win.show()

        app.exec_()


def analysisApp():

    from tormenta.analysis import analysis

    app = QtGui.QApplication([])

    win = analysis.AnalysisWidget()
    win.show()

    app.exec_()
