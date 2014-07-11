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
def generic_gaussian(xy, bkg, amp, x0, y0, sx, sy, theta):

    x, y = xy

    x0 = float(x0)
    y0 = float(y0)

    a = np.cos(theta)**2 / (2 * sx**2) + np.sin(theta)**2 / (2 * sy**2)
    b = - np.sin(2 * theta) / (4 * sx**2) + np.sin(2 * theta) / (4 * sy**2)
    c = np.sin(theta)**2 / (2*sx**2) + np.cos(theta)**2 / (2 * sy**2)

    g = bkg + amp * np.exp(- (a * (x - x0)**2 +
                           2 * b * (x - x0) * (y - y0) +
                           c * (y - y0)**2))
    return g.ravel()


def simmetric_gaussian(xy, bkg, amp, x0, y0, s):

    x, y = xy

    x0 = float(x0)
    y0 = float(y0)

    g = bkg + amp * np.exp(-(((x0 - x)/s)**2 + ((y0 - y)/s)**2) * 2)

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

    data1 = data1 - np.median(data1)
    data2 = data2 - np.median(data2)

    # Correlation calculation and cropping
    correlation = fftconvolve(data1, data2[::-1, ::-1], mode="same")
    imax = np.unravel_index(correlation.argmax(), correlation.shape)
    crop_l = 5
    crop_corr = correlation[imax[0] - crop_l:imax[0] + crop_l + 1,
                            imax[1] - crop_l:imax[1] + crop_l + 1]
#    crop_corr -= correlation.mean()

    # Fitting
    x = np.arange(2 * crop_l + 1)
    y = np.arange(2 * crop_l + 1)
    x, y = np.meshgrid(x, y)
    guess = np.concatenate(([crop_corr.min(),
                             crop_corr.max() - crop_corr.min()],
                            parameters(crop_corr)))
    popt, pcov = opt.curve_fit(generic_gaussian, (x, y),
                               crop_corr.ravel(), p0=guess)
#    guess = np.concatenate(([crop_corr.min(),
#                             crop_corr.max() - crop_corr.min()],
#                            parameters(crop_corr)[:3]))
#    popt, pcov = opt.curve_fit(simmetric_gaussian, (x, y),
#                               crop_corr.ravel(), p0=guess)

    print('guess', guess)
    print('popt', popt)

    # Plots
    data_fitted = generic_gaussian((x, y), *popt)
#    data_fitted = simmetric_gaussian((x, y), *popt)
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    corr_plot = ax.imshow(crop_corr, cmap=plt.cm.jet, origin='bottom',
                          extent=(x.min(), x.max(), y.min(), y.max()),
                          interpolation='None')
    fig.colorbar(corr_plot)
    ax.contour(x, y, data_fitted.reshape(2 * crop_l + 1, 2 * crop_l), 8,
               colors='w')

    # Drift calculation
    init = np.array(data1.shape) / 2
    drift = [imax[0] - crop_l + popt[2], imax[1] - crop_l + popt[3]] - init

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

scale = 133/40

nbins = np.ceil(np.array([253, 239]) * scale)
#histos = np.array([make_histo(p, nbins) for p in paths])

xl, yl = get_i3_results(paths[0])
for p in np.arange(1, len(paths)):
    xn, yn = get_i3_results(paths[p])
    xl, yl = np.hstack((xl, xn)), np.hstack((yl, yn))

n_locs = len(xl)
xl, yl = chunker(xl, 110000), chunker(yl, 110000)
histos = np.array([np.histogram2d(y, x, bins=nbins)[0]
                  for x, y in zip(xl, yl)])

xt, yt = drift_track(histos)
xt, yt = xt / scale, yt / scale
print(xt, yt)

x_loc, y_loc = xl[0], yl[0]
for p in np.arange(1, len(paths)):
    x_loc = np.hstack((x_loc, xl[p] + xt[p]))
    y_loc = np.hstack((y_loc, yl[p] + yt[p]))

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(17.0, 7.0))
gs = gridspec.GridSpec(2, 2)

ax00 = plt.subplot(gs[0, 0])
img00 = ax00.imshow(histos[0], interpolation='none', vmax=10)
plt.xlabel('data0')
fig.colorbar(img00, ax=ax00)

ax01 = plt.subplot(gs[0, 1])
img01 = ax01.imshow(histos[1], interpolation='none', vmax=10)
plt.xlabel('data1')
fig.colorbar(img01, ax=ax01)

ax10 = plt.subplot(gs[1, 0])
img10 = ax10.imshow(histos[0] + histos[1], interpolation='none', vmax=10)
plt.xlabel('data0 + data1')
fig.colorbar(img10, ax=ax10)

x_loc, y_loc = xl[0], yl[0]
x_loc = np.hstack((x_loc, xl[1] + xt[1]))
y_loc = np.hstack((y_loc, yl[1] + yt[1]))
H = np.histogram2d(y_loc, x_loc, bins=nbins)[0]
ax11 = plt.subplot(gs[1, 1])
img11 = ax11.imshow(H, interpolation='none', vmax=10)
plt.xlabel('corrected')
fig.colorbar(img11, ax=ax11)

# LOCALIZATION BINNING
#H, xedges, yedges = np.histogram2d(y_loc, x_loc, bins=nbins)
#extent = [yedges[0], yedges[-1],  xedges[0], xedges[-1]]
#
## scipy.misc.imsave('out.png', H)
#
#plt.imshow(H, extent=extent, cmap='gray', vmax=10, interpolation='none')
#plt.xlabel('$\mu$m')
#plt.ylabel('$\mu$m')
#plt.colorbar()


def compare(path0, path1):

    import matplotlib.gridspec as gridspec

    nbins = np.ceil(np.array([253, 239]) * 133/40)
    data0 = make_histo(path0, nbins)
    data1 = make_histo(path1, nbins)


    fig = plt.figure(figsize=(17.0, 7.0))
    gs = gridspec.GridSpec(2, 2)

    ax00 = plt.subplot(gs[0, 0])
    img00 = ax00.imshow(data0, interpolation='none')
    plt.xlabel('data0')
    fig.colorbar(img00, ax=ax00)

    ax01 = plt.subplot(gs[0, 1])
    img01 = ax01.imshow(data1, interpolation='none')
    plt.xlabel('data1')
    fig.colorbar(img01, ax=ax01)

    ax0 = plt.subplot(gs[1, 0])
    summed = data0 + data1
    img = ax0.imshow(summed, interpolation='none')
    plt.xlabel('Summed')
    fig.colorbar(img, ax=ax0)
    ax0.axis('image')

    # Correlation calculation and cropping
    correlation = fftconvolve(data0, data1[::-1, ::-1], mode="same")
    correlation = correlation - np.median(correlation)
    imax = np.unravel_index(correlation.argmax(), correlation.shape)
    crop_l = 12
    crop_corr = correlation[imax[0] - crop_l:imax[0] + crop_l,
                            imax[1] - crop_l:imax[1] + crop_l]
    crop_corr -= correlation.mean()

    # Fitting
    x = np.arange(2 * crop_l)
    y = np.arange(2 * crop_l)
    x, y = np.meshgrid(x, y)
    guess = np.concatenate(([crop_corr.min(),
                             crop_corr.max() - crop_corr.min()],
                            parameters(crop_corr)))
    popt, pcov = opt.curve_fit(generic_gaussian, (x, y),
                               crop_corr.ravel(), p0=guess)

    # Drift calculation
    init = np.array(data1.shape) / 2
    drift = [imax[0] - crop_l + popt[1], imax[1] - crop_l + popt[2]] - init
    dx, dy = drift[1]*40/133, drift[0]*40/133
    print('dx, dy', drift)

#    dx, dy = drift(data0, data1)
    x0, y0 = get_i3_results(path0)
    x1, y1 = get_i3_results(path1)
    x_loc, y_loc = np.hstack((x0, x1 + dx)), np.hstack((y0, y1 + dy))

    ax1 = plt.subplot(gs[1, 1])
    hc = np.histogram2d(y_loc, x_loc, bins=nbins, normed=True)[0]
    img1 = ax1.imshow(hc, interpolation='None')
    fig.colorbar(img1, ax=ax1)
    plt.xlabel('Corrected')
    ax1.axis('image')

    # Correlation plots
    data_fitted = generic_gaussian((x, y), *popt)
    fig, ax = plt.subplots(1, 1)
    ax.hold(True)
    corr_plot = ax.imshow(crop_corr, cmap=plt.cm.jet, origin='bottom',
                          extent=(x.min(), x.max(), y.min(), y.max()),
                          interpolation='None')
    fig.colorbar(corr_plot)
    ax.contour(x, y, data_fitted.reshape(2 * crop_l, 2 * crop_l), 8,
               colors='w')
