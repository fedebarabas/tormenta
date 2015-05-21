#!/usr/bin/python

import sa_library.readinsight3 as readinsight3
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


def get_i3_results(filename):

    i3_data_in = readinsight3.loadI3GoodOnly(filename)
    x_locs = i3_data_in['xc']
    y_locs = i3_data_in['yc']
#    chi2 = i3_data_in['i']
#    fit_sigma = 0.5 * i3_data_in['w']     # 0.5 factor to turn the width of
#    n_photons = i3_data_in['a']           # the gaussian into a sigma

    return x_locs, y_locs


if __name__ == '__main__':

    shape = np.array([253, 239])
    px_size = 133
    x_locs, y_locs = get_i3_results(sys.argv[1])
    shape = px_size * shape         # shape conversion to um
    render_px = 20

    # LOCALIZATION BINNING
    H, xedges, yedges = np.histogram2d(y_locs, x_locs,
                                       bins=np.ceil(shape * px_size/render_px))
    extent = [yedges[0], yedges[-1],  xedges[0], xedges[-1]]
    # Herror, Herror_bins = np.histogram(fit_error, bins=100)

    scipy.misc.imsave('out.png', H)

    plt.imshow(H, extent=extent, cmap='gray', vmax=10, interpolation='none')
    plt.xlabel('$\mu$m')
    plt.ylabel('$\mu$m')
    plt.colorbar()

#    i3_data_in = readinsight3.loadI3GoodOnly(sys.argv[1])

    # FIGURE CREATION
#    fig = plt.figure(figsize=(17.0, 7.0))
#    gs = gridspec.GridSpec(3, 2)

    # PLOT THE RESULTING IMAGE
#    ax1 = plt.subplot(gs[:, 0])
#    img = ax1.imshow(H, extent=extent, cmap='gray', vmax=10,
#                     interpolation='none')

#    fig.colorbar(img, ax=ax1)
#    ax1.axis('image')

    # PROFILE ATTEMPT (NOT WORKING CORRECTLY)
    # punto1 = find_nearest(xedges,4.5)[1],find_nearest(yedges,10)[1]
    # punto2 = find_nearest(xedges,5.2)[1],find_nearest(yedges,8.6)[1]
    # length = int(np.hypot(punto2[0]-punto1[0], punto2[1]-punto1[1]))
    # length = 1000000
    # lin_x, lin_y = np.linspace(punto1[0], punto2[0], length),
    #                np.linspace(punto1[1], punto2[1], length)
    # profile = H[lin_x.astype(np.int), lin_y.astype(np.int)]
    # # profile = scipy.ndimage.map_coordinates(H, np.vstack((lin_x,lin_y)))
    # axes[0].plot([punto1[0], punto2[0]], [punto1[1], punto2[1]], 'ro-')
    # ax1.plot([4.5, 5.2], [10, 8.6], 'ro-')

#    # CHI^2 OF THE FIT
#    ax2 = plt.subplot(gs[0, 1])
#    plt.hist(chi2, bins=np.arange(0, 200))
#    plt.xlabel('$\chi ^2 $ of the fit ')
#    plt.grid(True)
#
#    # SIGMA DETERMINED BY THE FIT
#    ax3 = plt.subplot(gs[1, 1])
#    plt.hist(fit_sigma, bins=np.arange(50, 300))
#    plt.xlabel('$\sigma $ of the gaussian [nm]')
#    plt.grid(True)
#
#    # PHOTONS PER LOCALIZATION
#    ax4 = plt.subplot(gs[2, 1])
#    plt.hist(n_photons, bins=np.arange(0, 3000), histtype='step')
#    plt.xlabel('Number of photons per localization')
#    plt.grid(True)
#
#    plt.tight_layout()
#    plt.show()
