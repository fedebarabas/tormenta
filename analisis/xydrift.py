import numpy as np
from scipy.signal import fftconvolve
import scipy.optimize as opt

# with open("d1.raw", 'rb') as d1:
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
def generic_gaussian(xy, amp, x0, y0, sx, sy, theta):

    x, y = xy

    x0 = float(x0)
    y0 = float(y0)

    a = np.cos(theta)**2 / (2 * sx**2) + np.sin(theta)**2 / (2 * sy**2)
    b = - np.sin(2 * theta) / (4 * sx**2) + np.sin(2 * theta) / (4 * sy**2)
    c = np.sin(theta)**2 / (2*sx**2) + np.cos(theta)**2 / (2 * sy**2)

    g = amp * np.exp(- (a * (x - x0)**2 +
                        2 * b * (x - x0) * (y - y0) +
                        c * (y - y0)**2))
    return g.ravel()


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()


def parameters(data):

    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x0 = m10 / data_sum
    y0 = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x0 * m01) / data_sum
    sx = (raw_moment(data, 2, 0) - x0 * m10) / data_sum
    sy = (raw_moment(data, 0, 2) - y0 * m01) / data_sum
    theta = 0.5 * np.arctan(2 * u11 / (sx - sy))

    return np.array([x0, y0, sx, sy, theta])


def drift(data1, data2):

    # Correlation calculation and cropping
    correlation = fftconvolve(data1, data2[::-1, ::-1], mode="same")
    imax = np.unravel_index(correlation.argmax(), correlation.shape)
    crop_l = 3
    crop_corr = correlation[imax[0] - crop_l:imax[0] + crop_l,
                            imax[1] - crop_l:imax[1] + crop_l]
    crop_corr -= correlation.mean()

    # Fitting
    x = np.arange(2 * crop_l)
    y = np.arange(2 * crop_l)
    x, y = np.meshgrid(x, y)
    guess = np.concatenate(([crop_corr.max()], parameters(crop_corr)))
    popt, pcov = opt.curve_fit(generic_gaussian, (x, y),
                               crop_corr.ravel(), p0=guess)

    # Plots
    data_fitted = generic_gaussian((x, y), *popt)
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    ax.imshow(crop_corr, cmap=plt.cm.jet, origin='bottom',
              extent=(x.min(), x.max(), y.min(), y.max()),
              interpolation='None')
    ax.contour(x, y, data_fitted.reshape(2 * crop_l, 2 * crop_l), 8,
               colors='w')

    # Drift calculation
    init = np.array(data1.shape) / 2
    drift = [imax[0] - crop_l + popt[1], imax[1] - crop_l + popt[2]] - init

    return drift[1], drift[0]


def drift_track(data):

    n = len(data)
    x, y = np.zeros(n), np.zeros(n)
    for i in np.arange(1, n):
        x[i], y[i] = drift(data[0], data[i])

    return x, y


import matplotlib.pyplot as plt

# Data loading
# folder = r'/home/federico/data/CM1/2014-06-17 - pngs drift/'
# file1 = '02b1t30fr100px40.png'
# file2 = '02b2t30fr100px40.png'
# data1 = Image.open(folder + file1)
# data2 = Image.open(folder + file2)
# data1 = np.asarray(data1)
# data2 = np.asarray(data2)
#
# print(drift(data1, data2))


def make_histo(path, nbins):
    xl, yl = get_i3_results(path)
    H, xedges, yedges = np.histogram2d(yl, xl, bins=nbins, normed=True)
    return H


def chunker(seq, size):
    return np.array([seq[pos:pos + size] for pos in range(0, len(seq), size)])


import os
os.chdir(r'/home/federico/codigo/python/tormenta/analisis/')
from get_i3_results import get_i3_results

folder = r'/home/federico/data/CM1/FedeFuentes/02/'
results = ['002sat30fra200.bin', '002sb1t30fra100.bin', '002sb2t30fra100.bin',
           '002sb3t30fra100.bin']
paths = [folder + r for r in results]

nbins = np.ceil(np.array([253, 239]) * 133/40)
#histos = np.array([make_histo(p, nbins) for p in paths])

xl, yl = get_i3_results(paths[0])
for p in np.arange(1, len(paths)):
    xn, yn = get_i3_results(paths[p])
    xl, yl = np.hstack((xl, xn)), np.hstack((yl, yn))

n_locs = len(xl)
xl, yl = chunker(xl, 100000), chunker(yl, 100000)
histos = np.array([np.histogram2d(y, x, bins=nbins)[0]
                  for x, y in zip(xl, yl)])


xt, yt = drift_track(histos)
xt, yt = xt * 40/133, yt * 40/133
print(xt, yt)

x_loc, y_loc = xl[0], yl[0]
for p in np.arange(1, len(paths)):
    x_loc = np.hstack((x_loc, xl[p] + xt[p]))
    y_loc = np.hstack((y_loc, yl[p] + yt[p]))


#xl, yl = get_i3_results(paths[0])
#for p in np.arange(1, len(paths)):
#    xn, yn = get_i3_results(paths[p])
##    xl, yl = np.hstack((xl, xn)), np.hstack((yl, yn))
#    xl, yl = np.hstack((xl, xn + xt[p])), np.hstack((yl, yn + yt[p]))

# LOCALIZATION BINNING
H, xedges, yedges = np.histogram2d(y_loc, x_loc, bins=nbins)
extent = [yedges[0], yedges[-1],  xedges[0], xedges[-1]]

# scipy.misc.imsave('out.png', H)

plt.imshow(H, extent=extent, cmap='gray', vmax=10, interpolation='none')
plt.xlabel('$\mu$m')
plt.ylabel('$\mu$m')
plt.colorbar()
