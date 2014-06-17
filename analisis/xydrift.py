#import library.imagecorrelation as imagecorrelation
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.signal import fftconvolve

#with open("d1.raw", 'rb') as d1:
#    with open("d2.raw", 'rb') as d2:
#        shape = (200,200)           # height,width
#        datatype = np.dtype('uint16')
#        data1 = np.fromfile(d1, dtype=datatype).reshape(shape)
#        data2 = np.fromfile(d2, dtype=datatype).reshape(shape)
#        # correlation = scipy.signal.fftconvolve(data1, data2[::-1, ::-1], mode="same")
#        [corr, dx, dy, xy_success] = imagecorrelation.xyOffset(data1, data2, 2, center = [100,100])
#        print dx,dy,xy_success
#        # imshow(correlation,cmap='gray',interpolation='none')       # Would you like to plot one of them?
#        # plt.colorbar()
#        # show()

from PIL import Image

folder = r'/home/federico/data/CM1/2014-06-17 - pngs drift/'
file1 = '02b1t30fr100px40.png'
file2 = '02b2t30fr100px40.png'
data1 = Image.open(folder + file1)
data2 = Image.open(folder + file2)

data1 = np.asarray(data1)
data2 = np.asarray(data2)
correlation = fftconvolve(data1, data2[::-1, ::-1], mode="same")
imax = unravel_index(correlation.argmax(), correlation.shape)
crop_corr = correlation[imax[0] - 20:imax[0] + 20, imax[1] - 20:imax[1] + 20]
crop_corr -= correlation.mean()
