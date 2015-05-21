# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 15:49:02 2015

@author: luciano
"""

import numpy as np
import matplotlib.pyplot as plt

rawData = np.loadtxt('filename_focusdata')
setPoint = rawData[0]
plt.plot(rawData[2], rawData[1], 'b-', rawData[2], setPoint, 'r-')

mean = np.mean(rawData[1])
std_dev = np.std(rawData[1])
max_dev = np.max(np.abs(np.array(rawData[1]) - setPoint))
print("mean = {}; std dev = {}; max dev = {}".format(mean, std_dev, max_dev))
