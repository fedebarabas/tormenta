#import library.imagecorrelation as imagecorrelation
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.signal import fftconvolve
import scipy.optimize as opt
from scipy.ndimage.measurements import center_of_mass as cm

#with open("d1.raw", 'rb') as d1:
#    with open("d2.raw", 'rb') as d2:
#        shape = (200,200)           # height,width
#        datatype = np.dtype('uint16')
#        data1 = np.fromfile(d1, dtype=datatype).reshape(shape)
#        data2 = np.fromfile(d2, dtype=datatype).reshape(shape)
#        correlation = scipy.signal.fftconvolve(data1,
#                                               data2[::-1, ::-1], mode="same")
#        print dx,dy,xy_success
#        Would you like to plot one of them?
#        imshow(correlation,cmap='gray',interpolation='none')

#        # plt.colorbar()
#        # show()


# Generic gaussian definition taken from
# http://en.wikipedia.org/wiki/Gaussian_function
def generic_gaussian((x, y), amp, a, b, c, x0, y0):
    gauss = amp * np.exp(- (a * (x - x0)**2 +
                            2 * b * (x - x0) * (y - y0) +
                            c * (y - y0)**2))
    return gauss.ravel()


from PIL import Image

# Data loading
folder = r'/home/federico/data/CM1/2014-06-17 - pngs drift/'
file1 = '02b1t30fr100px40.png'
file2 = '02b2t30fr100px40.png'
data1 = Image.open(folder + file1)
data2 = Image.open(folder + file2)
data1 = np.asarray(data1)
data2 = np.asarray(data2)

correlation = fftconvolve(data1, data2[::-1, ::-1], mode="same")
imax = unravel_index(correlation.argmax(), correlation.shape)
crop_l = 10

crop_corr = correlation[imax[0] - crop_l:imax[0] + crop_l,
                        imax[1] - crop_l:imax[1] + crop_l]
crop_corr -= correlation.mean()

x = np.arange(2 * crop_l)
y = np.arange(2 * crop_l)
x, y = np.meshgrid(x, y)

guess = (crop_corr.max(), 1, 1, 1) + cm(crop_corr)
popt, pcov = opt.curve_fit(generic_gaussian, (x, y),
                           crop_corr.ravel(), p0=guess)

data_fitted = generic_gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.hold(True)
ax.imshow(crop_corr, cmap=plt.cm.jet, origin='bottom',
          extent=(x.min(), x.max(), y.min(), y.max()), interpolation='None')
ax.contour(x, y, data_fitted.reshape(2 * crop_l, 2 * crop_l), 8, colors='w')
