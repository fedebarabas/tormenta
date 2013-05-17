import Image
import numpy as np
import scipy as sc 
from pylab import * 
import array 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2   # OpenCV bindings for Python
import time

# OPEN RAW IMAGE
t0 = time.clock()
with open("test.raw", 'rb') as fileobj:
    shape = (1363,185,197)          # (#frames,height,width)
    datatype = np.dtype('uint16')   # 16-bit unsigned little-endian byte order
    data = np.fromfile(fileobj,dtype=datatype).reshape(shape)
    dt = time.clock() - t0
    print dt, "s loading stack of images"
    print round(1000000*dt/shape[0],1), "us per image frame"
   
    # IMAGE BLURRING FOR NOISE REMOVAL
    t0 = time.clock()
    data_blurred = [cv2.blur(data[i],(2,2)) for i in np.arange(shape[0])]
    dt = time.clock() - t0
    print dt, "s blurring stack"
    print round(1000000*dt/shape[0],1), "us blurring per image frame" 

    # imshow(data_blurred[0],cmap='gray',interpolation='none')       # Would you like to plot one of them?
    # plt.colorbar()
    # show()
   
    

    flag, dsds = cv2.threshold(data_blurred[0], 500, 255, cv2.THRESH_BINARY)
    
    # data_binary = [cv2.threshold(data_blurred[i], 500, 255, cv2.THRESH_BINARY) for i in np.arange(shape[0])]

