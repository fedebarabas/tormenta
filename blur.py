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
   
# OPEN TIFF IMAGE
# with Image.open('test.tif') as data   # not working
    
    # FIGURE DEFINITION
    fig = plt.figure(figsize=(13,10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.5,1])

    feta = data[0,:,:]      # take first frame 
    
    ax1 = fig.add_subplot(gs[0,0])
    imshow(feta,cmap='gray',interpolation='none') 
    plt.colorbar()
   
    # FRAME HISTOGRAM
    ax2 = fig.add_subplot(gs[0,1])
    plt.hist(feta.flatten(), bins=60)
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
  
    # SMOOTH FILTERING
    ax3 = fig.add_subplot(gs[1,0])
    out = cv2.blur(feta, (2, 2))  # BLUR = SMOOTH BY AVERAGE
    imshow(out,cmap='gray',interpolation='none')
    plt.colorbar()
    ax4 = fig.add_subplot(gs[1,1])
    plt.hist(out.flatten(), bins=60)    
    plt.title("Histogram")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # SHOW PLOTS
    plt.show()
    gs.tight_layout(fig)
