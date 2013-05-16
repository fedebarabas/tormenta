import Image
import numpy as np
import scipy as sc 
from pylab import * 
import array 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2   # OpenCV bindings for Python

# OPEN RAW IMAGE
with open("test.raw", 'rb') as fileobj:
    shape = (1363,185,197)          # (#frames,height,width)
    datatype = np.dtype('uint16')   # 16-bit unsigned little-endian byte order
    data = np.fromfile(fileobj,dtype=datatype).reshape(shape)
   
    for i in np.arange(len(data)):
        data_blurred[i] = cv2.blur(data[i], (2, 2))  # BLUR = SMOOTH BY AVERAGE
