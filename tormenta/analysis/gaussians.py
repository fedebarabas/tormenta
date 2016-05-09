# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:41:17 2016

@author: Federico Barabas

Refs:
https://gist.github.com/andrewgiessel/6122739
"""

import numpy as np
from scipy import optimize


class twoDSymmGaussian():

    def __init__(self, data):
        self.fit(data)

    def function(self, xdata, A, xo, yo, sigma, offset):
        (x, y) = xdata
        xo = float(xo)
        yo = float(yo)
        c = 2*sigma**2
        g = offset + A*np.exp(- ((x-xo)**2 + (y-yo)**2)/c)
        return g.ravel()

    def moments(self, data):
        """Returns (height, x, y, sigma, offset)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        offset = data.min()
        data -= offset
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, (width_x + width_y)/2, offset

    def fit(self, data):

        # Create x and y indices
        x = np.arange(0, data.shape[0], dtype=float)
        y = np.arange(0, data.shape[1], dtype=float)
        x, y = np.meshgrid(x, y)

        initial = self.moments(data)

        popt, pcov = optimize.curve_fit(self.function, (x, y), data.ravel(),
                                        p0=initial)
        self.popt = popt
        self.epopt = np.sqrt([pcov[i, i] for i in np.arange(pcov.shape[0])])


def twoDGaussian(xdata, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp(- (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) +
                                  c*((y-yo)**2)))
    return g.ravel()


def moments(data):
    """Returns (height, x, y, width_x, width_y, offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    offset = data.min()
    data -= offset
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y, offset
