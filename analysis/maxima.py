# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:23:40 2014

@author: fbaraba
"""

import warnings
warnings.filterwarnings("error")

import numpy as np

from scipy.special import erf
from scipy.optimize import minimize
from scipy.ndimage import label
from scipy.ndimage.filters import convolve, maximum_filter
from scipy.ndimage.measurements import maximum_position, center_of_mass

import analysis.tools as tools


# data-type definitions
def fit_par(fit_model):
    if fit_model is '2d':
        return [('amplitude_fit', float), ('fit_x', float), ('fit_y', float),
                ('background_fit', float)]


def results_dt(fit_parameters):
    parameters = [('frame', int), ('maxima_x', int), ('maxima_y', int),
                  ('photons', float), ('sharpness', float),
                  ('roundness', float), ('brightness', float)]
    return np.dtype(parameters + fit_parameters)


class Maxima():
    """ Class defined as the local maxima in an image frame. """

    def __init__(self, image, fit_par=None, dt=0, fw=None, win_size=None,
                 kernel=None, xkernel=None, bkg_image=None):
        self.image = image
        self.bkg_image = bkg_image

        # Noise removal by convolving with a null sum gaussian. Its FWHM
        # has to match the one of the objects we want to detect.
        try:
            self.fwhm = fw
            self.win_size = win_size
            self.kernel = kernel
            self.xkernel = xkernel
            self.image_conv = convolve(self.image.astype(float), self.kernel)
        except RuntimeError:
            # If the kernel is None, I assume all the args must be calculated
            self.fwhm = tools.get_fwhm(670, 1.42) / 120
            self.win_size = int(np.ceil(self.fwhm))
            self.kernel = tools.kernel(self.fwhm)
            self.xkernel = tools.xkernel(self.fwhm)
            self.image_conv = convolve(self.image.astype(float), self.kernel)

        self.fit_par = fit_par
        self.dt = dt

    def find_old(self, alpha=5):
        """Local maxima finding routine.
        Alpha is the amount of standard deviations used as a threshold of the
        local maxima search. Size is the semiwidth of the fitting window.
        Adapted from http://stackoverflow.com/questions/16842823/
                            peak-detection-in-a-noisy-2d-array
        """
        self.alpha = alpha

        # Image mask
        self.imageMask = np.zeros(self.image.shape, dtype=bool)

        self.mean = np.mean(self.image_conv)
        self.std = np.sqrt(np.mean((self.image_conv - self.mean)**2))
        self.threshold = self.alpha*self.std + self.mean

        # Estimate for the maximum number of maxima in a frame
        nMax = self.image.size // (2*self.win_size + 1)**2
        self.positions = np.zeros((nMax, 2), dtype=int)
        nPeak = 0

        while 1:
            k = np.argmax(np.ma.masked_array(self.image_conv, self.imageMask))

            # index juggling
            j, i = np.unravel_index(k, self.image.shape)
            if(self.image_conv[j, i] >= self.threshold):

                # Saving the peak
                self.positions[nPeak] = tuple([j, i])

                # this is the part that masks already-found maxima
                x = np.arange(i - self.win_size, i + self.win_size + 1,
                              dtype=np.int)
                y = np.arange(j - self.win_size, j + self.win_size + 1,
                              dtype=np.int)
                xv, yv = np.meshgrid(x, y)
                # the clip handles cases where the peak is near the image edge
                self.imageMask[yv.clip(0, self.image.shape[0] - 1),
                               xv.clip(0, self.image.shape[1] - 1)] = True
                nPeak += 1
            else:
                break

        if nPeak > 0:
            self.positions = self.positions[:nPeak]
            self.drop_overlapping()
            self.drop_border()

    def find(self, alpha=5):
        """
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when
        the pixel's value is the neighborhood maximum, 0 otherwise). Taken from
        http://stackoverflow.com/questions/9111711/
        get-coordinates-of-local-maxima-in-2d-array-above-certain-value
        """
        self.alpha = alpha

        image_max = maximum_filter(self.image_conv, self.win_size)
        maxima = (self.image_conv == image_max)

        self.mean = np.mean(self.image_conv)
        self.std = np.sqrt(np.mean((self.image_conv - self.mean)**2))
        self.threshold = self.alpha*self.std + self.mean

        diff = (image_max > self.threshold)
        maxima[diff == 0] = 0

        labeled, num_objects = label(maxima)
        if num_objects > 0:
            self.positions = maximum_position(self.image, labeled,
                                              range(1, num_objects + 1))
            self.positions = np.array(self.positions).astype(int)
            self.drop_overlapping()
            self.drop_border()
        else:
            self.positions = np.zeros((0, 2), dtype=int)

    def drop_overlapping(self):
        """Drop overlapping spots."""
        n = len(self.positions)
        if n > 1:
            self.positions = tools.dropOverlapping(self.positions,
                                                   2*self.win_size + 1)
            self.overlaps = n - len(self.positions)
        else:
            self.overlaps = 0

    def drop_border(self):
        """ Drop near-the-edge spots. """
        keep = ((self.positions[:, 0] < 126) & (self.positions[:, 0] > 1) &
                (self.positions[:, 1] < 126) & (self.positions[:, 1] > 1))
        self.positions = self.positions[keep]

    def getParameters(self):
        """Calculate the roundness, brightness, sharpness"""

        # Results storage
        try:
            self.results = np.zeros(len(self.positions), dtype=self.dt)
        except TypeError:
            self.fit_model = '2d'
            self.fit_par = fit_par(self.fit_model)
            self.dt = results_dt(self.fit_par)
            self.results = np.zeros(len(self.positions), dtype=self.dt)

        self.results['maxima_x'] = self.positions[:, 0]
        self.results['maxima_y'] = self.positions[:, 1]

        mask = np.zeros((2*self.win_size + 1, 2*self.win_size + 1), dtype=bool)
        mask[self.win_size, self.win_size] = True

        i = 0
        for maxx in self.positions:
            # tuples make indexing easier (see below)
            p = tuple(maxx)
            masked = np.ma.masked_array(self.radius(self.image, maxx), mask)

            # Sharpness
            sharp_norm = self.image_conv[p] * np.mean(masked)
            self.results['sharpness'][i] = 100*self.image[p]/sharp_norm
            # Roundness
            hx = np.dot(self.radius(self.image, maxx)[2, :], self.xkernel)
            hy = np.dot(self.radius(self.image, maxx)[:, 2], self.xkernel)
            self.results['roundness'][i] = 2 * (hy - hx) / (hy + hx)
            # Brightness
            bright_norm = self.alpha * self.std
            self.results['brightness'][i] = 2.5*np.log(self.image_conv[p] /
                                                       bright_norm)

            i += 1

    def area(self, image, n):
        """Returns the area around the local maximum number n."""
        coord = self.positions[n]
        x1 = coord[0] - self.win_size
        x2 = coord[0] + self.win_size + 1
        y1 = coord[1] - self.win_size
        y2 = coord[1] + self.win_size + 1
        return image[x1:x2, y1:y2]

    def radius(self, image, coord):
        """Returns the area around the entered point."""
        x1 = coord[0] - self.win_size
        x2 = coord[0] + self.win_size + 1
        y1 = coord[1] - self.win_size
        y2 = coord[1] + self.win_size + 1
        return image[x1:x2, y1:y2]

    def fit(self, fit_model='2d'):

        self.mean_psf = np.zeros(self.area(self.image, 0).shape)

        for i in np.arange(len(self.positions)):

            # Fit and store fitting results
            area = self.area(self.image, i)
            bkg = np.mean(self.area(self.bkg_image, i))
            fit = fit_area(area, self.fwhm, bkg)
            offset = self.positions[i] - self.win_size
            fit[1] += offset[0]
            fit[2] += offset[1]

            # Can I do this faster if fit_area returned a struct array? TRY IT!
            m = 0
            for par in self.fit_par:
                self.results[par[0]][i] = fit[m]
                m += 1

            # Background-sustracted measured PSF
            bkg_subtract = area - fit[-1]
            # photons from molecule calculation
            self.results['photons'][i] = np.sum(bkg_subtract)
            self.mean_psf += bkg_subtract / self.results['photons'][i]


# TODO: run calibration routine for better fwhm estimate
def fit_area(area, fwhm, bkg, fit_results=np.zeros(4), center=2,
             x=np.arange(5)):

    # First guess of parameters
    area_bkg = area - bkg
    A = 1.54*area_bkg[center, center]
    x0, y0 = center_of_mass(area_bkg)

    # TODO: get error of each parameter from the fit
    fit_results = minimize(logll, [A, x0, y0, bkg], args=(fwhm, area),
                           bounds=[(0, np.max(area)), (1, 4), (1, 4),
                                   (0, np.min(area))],
                           method='L-BFGS-B', jac=ll_jac).x
    return fit_results


def dexp(x0, sigma, x):
    a = (x - x0) / sigma
    b = a + 1/sigma
    return (np.exp(-a*a) - np.exp(-b*b))/(np.sqrt(np.pi)*sigma)


def derf(x0, sigma, x):
    """ Auxiliary  function. x, x0 and sigma are in px units. """
    a = (x - x0) / sigma
    return 0.5 * (erf(a + 1/sigma) - erf(a))


def derfs(x0, y0, sigma, xy):
    """ Auxiliary  function. x, x0 and sigma are in px units. """
    ax = (xy - x0) / sigma
    ay = (xy - y0) / sigma
    erfx = erf(ax + 1/sigma) - erf(ax)
    return 0.25 * erfx[:, np.newaxis] * (erf(ay + 1/sigma) - erf(ay))


def logll(parameters, *args, xy=np.arange(5)):
    """ Log-likelihood function for an area of size size**2 around a local
    maximum with respect with a 2d symmetric gaussian of A amplitude centered
    in (x0, y0) with full-width half maximum fwhm on top of a background bkg
    as the model PSF. x, x0 and sigma are in px units.
    """
    A, x0, y0, bkg = parameters
    fwhm, area = args

#    fwhm *= 0.5*(np.log(2))**(-1/2)
#    fwhm *= 0.6

    lambda_p = A * derfs(x0, y0, fwhm * 0.6, xy) + bkg
    return np.sum(lambda_p - area * np.log(lambda_p))


def ll_jac(parameters, *args, xy=np.arange(5), jac=np.zeros((4, 5, 5))):
    """ Jacobian of the log-likelihood function for an area of size size**2
    around a local maximum with respect with a 2d symmetric gaussian of A
    amplitude centered in (x0, y0) with full-width half maximum fwhm on top of
    a background bkg as the model PSF. x, x0 and sigma are in px units.
    Order of derivatives: A, x0, y0, bkg.
    """
    A, x0, y0, bkg = parameters
    fwhm, area = args
    fwhm *= 0.6

    derfx = derf(x0, fwhm, xy)
    derfy = derf(y0, fwhm, xy)

    # d-L/d(A)
    jac[0] = derfx[:, np.newaxis] * derfy
    # d-L/d(x0) y d-L/d(y0)
    jac[1] = dexp(x0, fwhm, xy)[:, np.newaxis] * derfy
    jac[2] = derfx[:, np.newaxis] * dexp(y0, fwhm, xy)
    jac[1:3] *= A
    # d-L/d(bkg)
    jac[3] = 1
    jac *= 1 - area/(A * jac[0] + bkg)

    return np.sum(jac, (1, 2))


def ll_hess_diag(params, *args, xy=np.arange(5), hess=np.zeros((4, 5, 5))):
    """ Diagonal of the Hessian matrix of the log-likelihood function for an
    area of size size**2 around a local maximum with respect with a 2d
    symmetric gaussian of A amplitude centered in (x0, y0) with full-width half
    maximum fwhm on top of a background bkg as the model PSF. x, x0 and sigma
    are in px units.
    Order of derivatives: A, x0, y0, bkg.
    """
    A, x0, y0, bkg = params
    fwhm, area, x = args
    fwhm *= 0.6

    derfx = derf(x0, fwhm, xy)[:, np.newaxis]
    derfy = derf(y0, fwhm, xy)

    # d2-L/d(A)2
    hess[0] = derfx


def ll_hess(params, *args, xy=np.arange(5), hess=np.zeros((4, 4, 5, 5))):
    """ Full Hessian matrix of the log-likelihood function for an area of size
    size**2 around a local maximum with respect with a 2d symmetric gaussian of
    A amplitude centered in (x0, y0) with full-width half maximum fwhm on top
    of a background bkg as the model PSF. x, x0 and sigma are in px units.
    Order of derivatives: A, x0, y0, bkg.
    """
    A, x0, y0, bkg = params
    fwhm, area, x = args
    fwhm *= 0.6

    derfx = derf(x0, fwhm, xy)
    derfy = derf(y0, fwhm, xy)


#def ll_hess(params, *args):
#
#    A, x0, y0, bkg = params
#    fwhm, area, x = args
#
##    x, y = np.arange(pico.shape[0]), np.arange(pico.shape[1])
#    xx, yy = np.mgrid[0:5, 0:5]
#
#    hess = np.zeros((4, 4))
#
#    cc = np.sqrt(np.pi)*fwhm/3.33333333333333
#    xf = 1.66666666666667*(-x0 + xx)/fwhm
#    xf1 = xf + 1.66666666666667/fwhm
#    yf = 1.66666666666667*(yy - y0)/fwhm
#    yf1 = yf + 1.66666666666667/fwhm
#
#    derfx = -erf(xf) + erf(xf1)
#    derfx2 = derfx**2
#    derfy = -erf(yf) + erf(yf1)
#    derfy2 = derfy**2
#
#    fwhm2 = 0.282743338823081*fwhm**2
#    fwhm5 = 5.555555555556/fwhm
#    fwhmA = 0.531736155271655*A*fwhm
#    lamb_a = fwhm2*derfx*derfy
#    lamb = A*lamb_a
#    lamb_g = lamb + bkg
#
#    hess33 = area/lamb_g**2
#    c2 = 0.150344855914456*A**2*fwhm**3*hess33
#    ff = -area/lamb_g + 1
#    c3 = 3.33333333333333*fwhmA*ff/fwhm
#    dexpx = -np.exp(-xf*xf) + np.exp(-xf1*xf1)
#
#    dexpxx = dexpx/cc
#    dexpy = np.exp(-yf1*yf1) - np.exp(-yf*yf)
#    dexpyy = dexpy/cc
#
#
##    diff(factor, bkg)
#    hess[3, 3] = np.sum(hess33)
#
##    diff(jac0, A)
#    hess[0, 0] = fwhm2**2*np.sum(hess33*derfx2*derfy2)
#
##    diff(jac0, x0)
#    hess[0, 1] = -fwhm2*np.sum(dexpxx*derfy*(hess33*lamb + ff))
#    hess[1, 0] = hess[0, 1]
#
##    diff(jac0, y0)
#    hess[0, 2] = -fwhm2*np.sum((A*fwhm2*derfx*derfy*hess33 + ff)*dexpyy*derfx)
#    hess[2, 0] = hess[0, 2]
#
##    diff(jac0, bkg)
#    hess[0, 3] = np.sum(hess33*lamb_a)
#    hess[3, 0] = hess[0, 3]
#
##    diff(jac1, x0)
#    hess110 = c2*dexpxx*dexpx*derfy2
#    hess111 = xf1*np.exp(-xf1*xf1) + fwhm5*(x0-xx)*np.exp(-xf*xf)
#    hess111 *= -1.77245385090552*A*fwhm*ff*derfy/fwhm
#    hess[1, 1] = np.sum(hess110 + hess111)
#
##    diff(jac1, y0)
#    hess[1, 2] = fwhmA*np.sum(dexpyy*dexpx*(hess33*lamb + ff))
#    hess[2, 1] = hess[1, 2]
#
##    diff(jac1, bkg)
#    hess[1, 3] = -fwhmA*np.sum(hess33*dexpx*derfy)
#    hess[3, 1] = hess[1, 3]
#
##    diff(jac2, y0)
#    hess220 = -c2*dexpyy*dexpy*derfx2
#    hess221 = yf1*np.exp(-yf1*yf1) + fwhm5*(yy+y0)*np.exp(-yf*yf)
#    hess221 *= -1.77245385090552*A*fwhm*ff*derfx/fwhm
#    hess[2, 2] = np.sum(hess220 + hess221)
#
#
##    diff(jac2, bkg)
#    hess[2, 3] = -fwhmA*np.sum(hess33*dexpy*derfx)
#    hess[3, 2] = hess[2, 3]
#
#    return hess


# if __name__ == "__main__":

#    import matplotlib.pyplot as plt
#    se = stack.Stack(r'/home/federico/data/20150706 Unsain/muestra43.hdf5')
#    se.localize_molecules(ran=(0,1000))
#    mm = maxima.Maxima(se.imageData[10], se.fwhm)
#    mm.find()
#    mm.getParameters()
#    mm.fit()
#    plt.imshow(mm.image, interpolation='None')
#    plt.autoscale(False)
#    plt.plot(mm.results['maxima_y'], mm.results['maxima_x'], 'ro')
#    plt.plot(mm.results['fit_y'], mm.results['fit_x'], 'bo')
