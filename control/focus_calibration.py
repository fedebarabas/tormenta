# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 14:50:09 2015

@author: luciano
"""

from lantz.drivers.labjack.t7 import T7
from lantz.drivers.prior.nanoscanz import NanoScanZ
import numpy as np
from lantz import Q_
import matplotlib.pyplot as plt


    """
    The calibration is made with 100 steps. In each step it takes the mean
    of 100 measurements and saves that data in signalData,
    also saves the position at which this signalData is gathered.
    Then, it moves 40nm (or whatever step you choose) and so on.
    Finally it displays the results in the console and on a graph.
    """

um = Q_(1, 'um')
nm = Q_(1, 'nm')

NanoZ = NanoScanZ(12)
placa = T7()
signalData = []
positionData = []
step = 40*nm

for i in range(100):


    instantSignal = []

    for j in range(100):


        instantSignal.append(placa.analog_in[0].magnitude)

    meanSignal = np.mean(np.array(instantSignal))
    signalData.append(meanSignal)
    positionData.append(NanoZ.position.magnitude)
    NanoZ.moveRelative(step)

calibration = np.polyfit(np.array(signalData), np.array(positionData), 1)

print(calibration)

calibPlot = np.polyval(calibration, np.array(signalData))
plt.plot(np.array(signalData), np.array(positionData),
         'r--', np.array(signalData), calibPlot, 'b-')
