# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:47:36 2013

@author: federico
"""

from __future__ import division, with_statement, print_function

import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.console
import numpy as np
import os

from pyqtgraph.dockarea import *

def openraw(filename, shape=None, datatype=np.dtype('uint16')):
    # 16-bit unsigned little-endian byte order

    fileobj = open(filename, 'rb')

    if shape==None:
        
        print('Shape not provided, loading it from inf file')
        rootname, ext = os.path.splitext(filename)
        inf_name = rootname + '.inf'
        inf_data = np.loadtxt(inf_name, dtype=str)
#       self.frame_rate = float(inf_data[22][inf_data[22].find('=') + 1:-1])
        n_frames = int(inf_data[29][inf_data[29].find('=') + 1:])
        frame_size = [int(inf_data[8][inf_data[8].find('=') + 1:]),
                      int(inf_data[8][inf_data[8].find('=') + 1:])]
        shape = (n_frames, frame_size[0], frame_size[1])
        print(shape)

    data = np.fromfile(fileobj, dtype=datatype).byteswap().reshape(shape)
    
    return data, shape

class Stack():
    
    def __init__(self, filename):

        self.image = Image.open(filename)
        self.size = self.image.size
        
        try:    
            while 1:
                self.image.seek(self.image.tell() + 1)
                
        except EOFError:
            pass
            
        self.nframes = self.image.tell()
        

    def frame(self, n):
        
        if n <= self.nframes:
            self.image.seek(n)
            data = np.array(self.image.getdata()).reshape(self.image.size[::-1])
            return np.transpose(data)
        
        else:
            print(n, "is greater than the number of frames of the stack")

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(1200, 700)
win.setWindowTitle('pyqtgraph example: dockarea')

## Create docks, place them into the window one at a time.
## Note that size arguments are only a suggestion; docks will still have to
## fill the entire dock area and obey the limits of their internal widgets.
d2 = Dock("Dock2 - Console", size=(500, 300))
d3 = Dock("Dock3", size=(500, 400))
d4 = Dock("Dock4 (tabbed) - Plot", size=(500, 200))
d5 = Dock("Dock5 - Image", size=(700, 500))
d6 = Dock("Dock6 (tabbed) - Plot", size=(500, 200))
area.addDock(d2, 'right')     
area.addDock(d3, 'top', d2)      
area.addDock(d4, 'above', d3)     
area.addDock(d5, 'left')  
area.addDock(d6, 'above', d3)
    
## Test ability to move docks programatically after they have been placed
#area.moveDock(d4, 'top', d2)     ## move d4 to top edge of d2
#area.moveDock(d6, 'above', d4)   ## move d6 to stack on top of d4
#area.moveDock(d5, 'top', d2)     ## move d5 to top edge of d2


## Add widgets into each dock

w2 = pg.console.ConsoleWidget()
d2.addWidget(w2)

## Hide title bar on dock 3
w3 = pg.PlotWidget(title="Plot inside dock with no title bar")
w3.plot(np.random.normal(size=100))
d3.addWidget(w3)

w4 = pg.PlotWidget(title="Dock 4 plot")
w4.plot(np.random.normal(size=100))
d4.addWidget(w4)

data = Stack('muestra.tif')

w5 = pg.ImageView()
#w5.setImage(data.frame(2))
d5.addWidget(w5)

w6 = pg.PlotWidget(title="Dock 6 plot")
w6.plot(np.random.normal(size=100))
d6.addWidget(w6)
    
win.show()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()