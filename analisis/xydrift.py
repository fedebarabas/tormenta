import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass
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


def raw_moment(data, iord, jord):
    nrows, ncols = data.shape
    y, x = np.mgrid[:nrows, :ncols]
    data = data * x**iord * y**jord
    return data.sum()


def gen_gauss_est(data):

    height = np.max(data) - np.median(data)
    bkg = np.median(data)

    data = data - np.min(data)

    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x0 = m10 / data_sum
    y0 = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x0 * m01) / data_sum
    sx = np.sqrt((raw_moment(data, 2, 0) - x0 * m10) / data_sum)
    sy = np.sqrt((raw_moment(data, 0, 2) - y0 * m01) / data_sum)
    theta = 0.5 * np.arctan(2 * u11 / (sx - sy))

    return bkg, height, x0, y0, sx, sy, theta


def sim_gauss_est(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """

    height = np.max(data) - np.median(data)
    bkg = np.median(data)

    data = data - np.min(data)

    total = data.sum()
    X, Y = np.indices(data.shape)
    x0 = (X*data).sum()/total
    y0 = (Y*data).sum()/total
    col = data[:, int(y0)]
    sx = np.sqrt(abs((np.arange(col.size) - y0)**2*col).sum()/col.sum())
    row = data[int(x0), :]
    sy = np.sqrt(abs((np.arange(row.size) - x0)**2*row).sum()/row.sum())

    return bkg, height, x0, y0, sx, sy


def simmetric_gaussian(bkg, height, x0, y0, sx, sy):
    """Returns a gaussian function with the given parameters"""
    sx = float(sx)
    sy = float(sy)
    return lambda x, y: bkg + height*np.exp(-(((x0 - x)/sx)**2
                                            + ((y0 - y)/sy)**2)/2)


# Generic gaussian definition taken from
# http://en.wikipedia.org/wiki/Gaussian_function
def generic_gaussian(bkg, amp, x0, y0, sx, sy, theta):
    x0 = float(x0)
    y0 = float(y0)

    a = np.cos(theta)**2 / (2 * sx**2) + np.sin(theta)**2 / (2 * sy**2)
    b = - np.sin(2 * theta) / (4 * sx**2) + np.sin(2 * theta) / (4 * sy**2)
    c = np.sin(theta)**2 / (2*sx**2) + np.cos(theta)**2 / (2 * sy**2)

    return lambda x, y: bkg + amp * np.exp(- (a * (x - x0)**2 +
                                           2 * b * (x - x0) * (y - y0) +
                                           c * (y - y0)**2))


def fit_LS(function, data, params):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    errorfunction = lambda p: np.ravel(function(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = opt.leastsq(errorfunction, params)
    return p


def drift(data1, data2, i):

    data1 = data1 - np.median(data1)
    data2 = data2 - np.median(data2)

    # Correlation calculation and cropping
    correlation = fftconvolve(data1, data2[::-1, ::-1], mode="same")
    imax = np.unravel_index(correlation.argmax(), correlation.shape)
    crop_l = 10
    crop_corr = correlation[imax[0] - crop_l:imax[0] + crop_l + 1,
                            imax[1] - crop_l:imax[1] + crop_l + 1]

    params_gen = fit_LS(generic_gaussian, crop_corr,
                        gen_gauss_est(crop_corr))
    params_sim = fit_LS(simmetric_gaussian, crop_corr,
                        sim_gauss_est(crop_corr))
    params_cm = center_of_mass(crop_corr - np.min(crop_corr))

    # Drift calculation
    init = np.floor(np.array(data1.shape) / 2)
    drift_gen = [imax[0] - crop_l + params_gen[2],
                 imax[1] - crop_l + params_gen[3]] - init
    drift_sim = [imax[0] - crop_l + params_sim[2],
                 imax[1] - crop_l + params_sim[3]] - init
    drift_cm = [imax[0] - crop_l + params_cm[0],
                imax[1] - crop_l + params_cm[1]] - init

    # Plots
    if i == 1:
        fit = generic_gaussian(*params_gen)
        fig, ax = plt.subplots(1, 1)
        ax.hold(True)
        corr_plot = ax.imshow(crop_corr, cmap=plt.cm.jet, origin='bottom',
                              interpolation='None')
        fig.colorbar(corr_plot)
        ax.contour(fit(*np.indices(crop_corr.shape)), cmap=plt.cm.copper)
        print()
        print(drift_gen)

    return (drift_gen[0], drift_gen[1],
            drift_sim[0], drift_sim[1],
            drift_cm[0], drift_cm[1])


def drift_track(data):

    n = len(data)
    x_gen, y_gen = np.zeros(n), np.zeros(n)
    x_sim, y_sim = np.zeros(n), np.zeros(n)
    x_cm, y_cm = np.zeros(n), np.zeros(n)
    for i in np.arange(1, n):
        (x_gen[i], y_gen[i],
         x_sim[i], y_sim[i], x_cm[i], y_cm[i]) = drift(data[0], data[i], i)

    return x_gen, y_gen, x_sim, y_sim, x_cm, y_cm


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
# histos = np.array([make_histo(p, nbins) for p in paths])

xl, yl = get_i3_results(paths[0])
for p in np.arange(1, len(paths)):
    xn, yn = get_i3_results(paths[p])
    xl, yl = np.hstack((xl, xn)), np.hstack((yl, yn))

n_locs = len(xl)
xl, yl = chunker(xl, 100000), chunker(yl, 100000)
histos = np.array([np.histogram2d(y, x, bins=nbins)[0]
                  for x, y in zip(xl, yl)])

tracks = drift_track(histos)
xt, yt = tracks[0], tracks[1]
print(xt[1], yt[1])
#xt, yt = xt / scale, yt / scale
#print(xt[1], yt[1])

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
