import library.imagecorrelation as imagecorrelation
import numpy as np
import matplotlib.pyplot as plt
from pylab import * 
import scipy

with open("d1.raw", 'rb') as d1:
    with open("d2.raw", 'rb') as d2:
        shape = (200,200)           # height,width
        datatype = np.dtype('uint16')
        data1 = np.fromfile(d1, dtype=datatype).reshape(shape)
        data2 = np.fromfile(d2, dtype=datatype).reshape(shape)
        # correlation = scipy.signal.fftconvolve(data1, data2[::-1, ::-1], mode="same")
        [corr, dx, dy, xy_success] = imagecorrelation.xyOffset(data1, data2, 2, center = [100,100])
        print dx,dy,xy_success
        # imshow(correlation,cmap='gray',interpolation='none')       # Would you like to plot one of them?
        # plt.colorbar()
        # show()

