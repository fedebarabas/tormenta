# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 17:42:38 2015

@author: federico
"""
import numpy as np


def cubeHelixLUT(start=0.5, hue=1.0, gamma=1.0, rots=-1.5):

    lut = np.zeros((256, 3), dtype=np.ubyte)

    for i in np.arange(256):

        fract = i/255.0
        angle = 2 * np.pi * (start/3.0 + 1.0 + rots*fract)
        fract = fract**gamma
        amp = hue*fract*(1-fract)/2.0

        r = fract + amp*(-0.14861*np.cos(angle) + 1.78277*np.sin(angle))
        g = fract + amp*(-0.29227*np.cos(angle) - 0.90649*np.sin(angle))
        b = fract + amp*(1.97294*np.cos(angle))

        # on sature les negatifs
        r = np.median([1, r, 0])
        g = np.median([1, g, 0])
        b = np.median([1, b, 0])

        lut[i] = [np.round(r*255), np.round(g*255), np.round(b*255)]

    return lut

import pyqtgraph as pg

# random image data
img = np.random.normal(size=(100, 100))

# GUI
win = pg.GraphicsWindow()
view = win.addViewBox()
view.setAspectLocked(True)
item = pg.ImageItem(img)
view.addItem(item)
item.setLookupTable(cubeHelixLUT())
item.setLevels([0, 1])

win.show()
