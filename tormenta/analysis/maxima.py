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

import tormenta.analysis.tools as tools


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
                 kernel=None, xkernel=None):
        self.image = image

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
                x = np.arange(i - self.win_size, i + self.win_size + 1, dtype=np.int)
                y = np.arange(j - self.win_size, j + self.win_size + 1, dtype=np.int)
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
            self.positions = np.array(self.positions)
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

        # Background estimation. Taking the mean counts of the molecule-free
        # area is good enough and ~10x faster than getting the mode
        # 215 µs vs 1.89 ms
        try:
            self.imageMask
        except:
            self.imageMask = np.zeros(self.image.shape, dtype=bool)
            for p in self.positions:
                self.imageMask[p[0] - self.win_size:p[0] + self.win_size + 1,
                               p[1] - self.win_size:p[1] + self.win_size + 1] = True

        self.imageMask[self.image == 0] = True
        self.bkg = np.mean(np.ma.masked_array(self.image, self.imageMask))

        # Results storage
        try:
            self.results = np.zeros(len(self.positions), dtype=self.dt)
        except TypeError:
            self.fit_model = '2d'
            self.fit_par = fit_par(self.fit_model)
            self.dt = results_dt(self.fit_par)

        self.results['maxima_px'] = self.positions

        mask = np.zeros((2*self.win_size + 1, 2*self.win_size + 1), dtype=bool)
        mask[self.win_size, self.win_size] = True

        i = 0
        for maxx in self.positions:
            # tuples make indexing easier (see below)
            p = tuple(maxx)
            masked = np.ma.masked_array(self.radius(maxx), mask)

            # Sharpness
            sharp_norm = self.image_conv[p] * np.mean(masked)
            self.results['sharpness'][i] = 100*self.image[p]/sharp_norm
            # Roundness
            hx = np.dot(self.radius(maxx)[2, :], self.xkernel)
            hy = np.dot(self.radius(maxx)[:, 2], self.xkernel)
            self.results['roundness'][i] = 2 * (hy - hx) / (hy + hx)
            # Brightness
            bright_norm = self.alpha * self.std
            self.results['brightness'][i] = 2.5*np.log(self.image_conv[p] /
                                                       bright_norm)

            i += 1

    def area(self, n):
        """Returns the area around the local maximum number n."""
        coord = self.positions[n]
        x1 = coord[0] - self.win_size
        x2 = coord[0] + self.win_size + 1
        y1 = coord[1] - self.win_size
        y2 = coord[1] + self.win_size + 1
        return self.image[x1:x2, y1:y2]

    def radius(self, coord):
        """Returns the area around the entered point."""
        x1 = coord[0] - self.win_size
        x2 = coord[0] + self.win_size + 1
        y1 = coord[1] - self.win_size
        y2 = coord[1] + self.win_size + 1
        return self.image[x1:x2, y1:y2]

    def fit(self, fit_model='2d'):

        self.mean_psf = np.zeros(self.area(0).shape)

        for i in np.arange(len(self.positions)):

            # Fit and store fitting results
            area = self.area(i)
            fit = fit_area(area, self.fwhm, self.bkg)
            offset = self.positions[i] - self.win_size - 0.5
            fit[1] += offset[1]
            fit[2] += offset[2]

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
def fit_area(area, fwhm, bkg):

    # First guess of parameters
    A = 1.54*(area[area.shape[0] // 2, area.shape[1] // 2] - bkg)
    x0, y0 = center_of_mass(area)
    minn = np.min(area)
    maxx = np.max(area)

    results = np.zeros((1,), dtype=np.dtype(fit_par('2d')))

    # TODO: get error of each parameter from the fit (see Powell?)
    # 1.75 is an ad hoc factor to minimize the number of iterations during fit
    try:
        fit_results = minimize(logll, [A, x0, y0, bkg],
                               bounds=[(0, maxx), (1, 4), (1, 4), (0, minn)],
                               args=(fwhm * 1.75, area, np.arange(len(area))),
                               method='L-BFGS-B', jac=ll_jac)
#        results = minimize(logll, [A, x0, y0, bkg],
#                          args=(fwhm * 1.75, area, np.arange(len(area))),
#                          method='Powell')

        results = fit_results.x

#    except RuntimeWarning:
        # TODO: store the warning in results
#        print('CACHED!')

    return results


def dexp(x, x0, sigma):
    return np.exp(-((x + 1 - x0)/sigma)**2) - np.exp(-((x - x0)/sigma)**2)


def derf(x, x0, sigma):
    """ Auxiliary  function. x, x0 and sigma are in px units. """
    return erf((x + 1 - x0) / sigma) - erf((x - x0) / sigma)


def derfs(xy, x0, y0, sigma):
    """ Auxiliary  function. x, x0 and sigma are in px units. """
    i = erf((xy + 1 - x0) / sigma) - erf((xy - x0) / sigma)
    j = erf((xy + 1 - y0) / sigma) - erf((xy - y0) / sigma)
    return i[:, np.newaxis] * j


def lambda_g(x0, y0, fwhm, vector, factor=0.09*np.pi, f2=0.6):
    """ Theoretical mean number of photons detected in an area of size size**2
    due to the emission of a molecule located in (x0, y0). The model PSF is
    a 2d symmetric gaussian of A amplitude with full-width half maximum fwhm.
    x, x0 and fwhm are in px units.
    """
#    fwhm *= 0.5*(np.log(2))**(-1/2)
#    fwhm *= 0.6
#    0.6*0.6*0.25 = 0.09

    return factor * fwhm**2 * derfs(vector, x0, y0, fwhm * f2)


def logll(parameters, *args):
    """ Log-likelihood function for an area of size size**2 around a local
    maximum with respect with a 2d symmetric gaussian of A amplitude centered
    in (x0, y0) with full-width half maximum fwhm on top of a background bkg
    as the model PSF. x, x0 and sigma are in px units.
    """
    A, x0, y0, bkg = parameters
    fwhm, area, vector = args

    lambda_p = A * lambda_g(x0, y0, fwhm, vector) + bkg
    return -np.sum(area * np.log(lambda_p) - lambda_p)


# TODO: working?
def ll_jac(parameters, *args):
    """ Jacobian of the log-likelihood function for an area of size size**2
    around a local maximum with respect with a 2d symmetric gaussian of A
    amplitude centered in (x0, y0) with full-width half maximum fwhm on top of
    a background bkg as the model PSF. x, x0 and sigma are in px units.
    Order of derivatives: A, x0, y0, bkg.
    """
    A, x0, y0, bkg = parameters
    fwhm, area, vector = args

#    fwhm *= 0.5*(np.log(2))**(-1/2)

    derfx = derf(vector, x0, fwhm*0.6)
    derfy = derf(vector, y0, fwhm*0.6)
    lambda1 = lambda_g(x0, y0, fwhm, vector)
    factor = 1 - area/(A * lambda1 + bkg)

    jac = np.zeros(4)
    # dL/d(A)
    # The derivative of lambda_g is lambda_g(A=1)
    jac[0] = np.sum(factor*lambda1)
    # dL/d(x0) y dL/d(y0)
    # 0.3 = 0.5*0.6
    jac12 = -0.3*A*fwhm*np.sqrt(np.pi)
    jac[1] = jac12*np.sum(dexp(vector, x0, fwhm * 0.6)[:, np.newaxis] *
                          derfy*factor)
    jac[2] = jac12*np.sum(dexp(vector, y0, fwhm * 0.6)[:, np.newaxis] *
                          derfx*factor)
    # dL/d(bkg)
    jac[3] = np.sum(factor)

    return jac


# def ll_hess(params, *args):
#
#    A, x0, y0, bkg = params
#    pico, F = args
#
#    x, y = np.arange(pico.shape[0]), np.arange(pico.shape[1])
#
#    erfi = derf(x, x0, F)
#    erfj = derf(y, y0, F)
#    erfij = erfi*erfj
#    expi = ex(x, x0 - 1, F) - ex(x, x0, F)
#    expj = ex(y, y0 - 1, F) - ex(y, y0, F)
#
#    hess = np.zeros((4, 4))
#
#    # All derivatives made with sympy
#
#    # expr.diff(A, A)
#    hess[0, 0] = - np.sum(0.616850275068085 * F**4 * pico * erfi**2 * erfj**2 /
#                          ((np.pi/4) * A * F**2 * (erfi * erfj + bkg)**2))
#
#    # expr.diff(A, x0)
#    hessi01 = F*expi*erfj*(-1.23370055013617*A*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2 + 1.5707963267949*pico/((np.pi/4)*A*F**2*erfij + bkg) - 1.5707963267949)/np.sqrt(np.pi)
#    hess[0, 1] = np.sum(hessi01)
#    hess[1, 0] = hess[0, 1]
#
#    # expr.diff(A, y0)
#    hess[0, 2] = np.sum(F*expj*(-erfi)*(-1.23370055013617*A*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2 + (np.pi/2)*pico/((np.pi/4)*A*F**2*erfij + bkg) - (np.pi/2))/np.sqrt(np.pi))
#    hess[2, 0] = hess[0, 2]
#
#    # expr.diff(A, bkg)
#    hessi03 = -(np.pi/4)*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2
#    hess[0, 3] = np.sum(hessi03)
#    hess[3, 0] = hess[0, 3]
#
#    # expr.diff(x0, x0)
#    hess[1, 1] = np.sum(A*erfj*(2.46740110027234*A*F**2*pico*expi**2*(-erfj)/(np.pi*((np.pi/4)*A*F**2*erfij + bkg)**2) - np.pi*pico*((x - x0)*ex(x, x0, F) - (x - x0 + 1)*ex(x, x0 - 1, F))/(np.sqrt(np.pi)*F*((np.pi/4)*A*F**2*erfij + bkg)) + (np.pi*(x - x0)*ex(x, x0, F) - np.pi*(x - x0 + 1)*ex(x, x0 - 1, F))/(np.sqrt(np.pi)*F)))
#
#    # expr.diff(x0, y0)
#    hess[1, 2] = np.sum(A*expi*expj*(-2.46740110027234*A*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2 + np.pi*pico/((np.pi/4)*A*F**2*erfij + bkg) - np.pi)/np.pi)
#    hess[2, 1] = hess[1, 2]
#
#    # expr.diff(x0, bkg)
#    hess[1, 3] = np.sum(-(np.pi/2)*A*F*pico*expi*(-erfj)/(np.sqrt(np.pi)*((np.pi/4)*A*F**2*erfij + bkg)**2))
#    hess[3, 1] = hess[1, 3]
#
#    # expr.diff(y0, y0)
#    hess[2, 2] = np.sum(A*(-erfi)*(-2.46740110027234*A*F**2*pico*expj**2*(-erfi)/(np.pi*((np.pi/4)*A*F**2*erfij + bkg)**2) - np.pi*pico*((y - y0)*ex(y, y0, F) - (y - y0 + 1)*ex(y, y0 -1, F))/(np.sqrt(np.pi)*F*((np.pi/4)*A*F**2*erfij + bkg)) + (np.pi*(y - y0)*ex(y, y0, F) - np.pi*(y - y0 + 1)*ex(y, y0 -1, F))/(np.sqrt(np.pi)*F)))
#
#    # expr.diff(y0, bkg)
#    hess[2, 3] = np.sum(-(np.pi/2)*A*F*pico*expj*(-erfi)/(np.sqrt(np.pi)*((np.pi/4)*A*F**2*erfij + bkg)**2))
#    hess[3, 2] = hess[2, 3]
#
#    # expr.diff(bkg, bkg)
#    hess[3, 3] = np.sum(-pico/((np.pi/4)*A*F**2*erfij + bkg)**2)
#
#    return hess

if __name__ == "__main__":

    import tormenta.analysis.stack as stack

    stack = stack.Stack()
    maxima = Maxima(stack.image[10], stack.fwhm)

#        plt.imshow(mm.image, interpolation='None')
#        plt.autoscale(False)
#        plt.plot(maxima[:, 1], maxima[:, 0], 'ro')
#        plt.plot(mm.positions[:, 1], mm.positions[:, 0], 'ro')
