# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:23:40 2014

@author: fbaraba
"""

import numpy as np

from scipy.special import erf
from scipy.optimize import minimize
from scipy.ndimage import label
from scipy.ndimage.filters import convolve, maximum_filter
from scipy.ndimage.measurements import maximum_position, center_of_mass

import tormenta.analysis.tools as tools


# data-type definitions
def results_dt(fit_model):
    if fit_model is '2d':
        fit_parameters = [('amplitude', float), ('maxima_fit', np.float, (2,)),
                          ('background', float)]
        n_fit_par = len(fit_parameters)
    parameters = [('frame', int), ('maxima_px', np.int, (2,)),
                  ('photons', float), ('sharpness', float),
                  ('roundness', float), ('brightness', float)]
    return n_fit_par, np.dtype(parameters + fit_parameters)


class Maxima():
    """ Class defined as the local maxima in an image frame. """

    def __init__(self, image, fw=None, kernel=None):
        self.image = image
        if fw is None:
            self.fwhm = tools.get_fwhm(670, 1.42) / 120
        else:
            self.fwhm = fw
        self.size = int(np.ceil(self.fwhm))
        if kernel is None:
            self.kernel = tools.kernel(self.fwhm)
        else:
            self.kernel = kernel

    def find_old(self, alpha=5):
        """Local maxima finding routine.
        Alpha is the amount of standard deviations used as a threshold of the
        local maxima search. Size is the semiwidth of the fitting window.
        Adapted from http://stackoverflow.com/questions/16842823/
                            peak-detection-in-a-noisy-2d-array
        """
        self.alpha = alpha

        # Noise removal by convolving with a null sum gaussian. Its FWHM
        # has to match the one of the objects we want to detect.
        self.image_conv = convolve(self.image.astype(float), self.kernel)

        # Image mask
        self.imageMask = np.zeros(self.image.shape, dtype=bool)

        self.mean = np.mean(self.image_conv)
        self.std = np.sqrt(np.mean((self.image_conv - self.mean)**2))
        self.threshold = self.alpha*self.std + self.mean

        # Estimate for the maximum number of maxima in a frame
        nMax = np.ceil(self.image.size / (2*self.size + 1)**2)
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
                x = np.arange(i - self.size, i + self.size + 1, dtype=np.int)
                y = np.arange(j - self.size, j + self.size + 1, dtype=np.int)
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

        # Noise removal by convolving with a null sum gaussian. Its FWHM
        # has to match the one of the objects we want to detect.
        self.image_conv = convolve(self.image.astype(float), self.kernel)

        image_max = maximum_filter(self.image_conv, self.size)
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


#        plt.imshow(mm.image, interpolation='None')
#        plt.autoscale(False)
#        plt.plot(maxima[:, 1], maxima[:, 0], 'ro')
#        plt.plot(mm.positions[:, 1], mm.positions[:, 0], 'ro')

    def drop_overlapping(self):
        """Drop overlapping spots."""
        n = len(self.positions)

        if n > 1:
            self.positions = tools.dropOverlapping(self.positions,
                                                   2*self.size + 1)
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

#        slices = ndimage.find_objects(labeled)

        # Background estimation. Taking the mean counts of the molecule-free
        # area is good enough and ~10x faster than getting the mode
        # 215 µs vs 1.89 ms
        try:
            self.imageMask
        except:
            self.imageMask = np.zeros(self.image.shape, dtype=bool)
            for p in self.positions:
                self.imageMask[p[0] - self.size:p[0] + self.size + 1,
                               p[1] - self.size:p[1] + self.size + 1] = True

        self.imageMask[self.image == 0] = True
        self.bkg = np.mean(np.ma.masked_array(self.image, self.imageMask))
        self.xkernel = tools.xkernel(self.fwhm)

        # Peak parameters
        roundness = np.zeros(len(self.positions))
        brightness = np.zeros(len(self.positions))

        sharpness = np.zeros(len(self.positions))
        mask = np.zeros((2*self.size + 1, 2*self.size + 1), dtype=bool)
        mask[self.size, self.size] = True

        for i in np.arange(len(self.positions)):
            # tuples make indexing easier (see below)
            p = tuple(self.positions[i])

            # Sharpness
            masked = np.ma.masked_array(self.area(i), mask)
            sharpness[i] = self.image[p] / (self.image_conv[p] * masked.mean())

            # Roundness
            hx = np.dot(self.area(i)[2, :], self.xkernel)
            hy = np.dot(self.area(i)[:, 2], self.xkernel)
            roundness[i] = 2 * (hy - hx) / (hy + hx)

            # Brightness
            brightness[i] = 2.5 * np.log(self.image_conv[p] /
                                         self.alpha*self.std)

        self.sharpness = sharpness
        self.roundness = roundness
        self.brightness = brightness

    def area(self, n):
        """Returns the area around the local maximum number n."""
        coord = self.positions[n]
        return self.image[coord[0] - self.size:coord[0] + self.size + 1,
                          coord[1] - self.size:coord[1] + self.size + 1]

    def radius(self, coord):
        """Returns the area around the entered point."""
        return self.image[coord[0] - self.size:coord[0] + self.size + 1,
                          coord[1] - self.size:coord[1] + self.size + 1]

    def fit(self, fit_model='2d'):

        npar, self.dt = results_dt(fit_model)

        self.results = np.zeros(len(self.positions), dtype=self.dt)
        self.mean_psf = np.zeros(self.area(0).shape)

        for i in np.arange(len(self.positions)):

            # Fit and store fitting results
            area = self.area(i)
            fit = fit_area(area, self.fwhm, self.bkg)
            fit[1] += self.positions[i] - self.size - 0.5
            for p in np.arange(npar):
                self.results[self.dt.names[-npar + p]][i] = fit[p]

            # Background-sustracted measured PSF
            bkg_subtract = area - fit[-1]
            # photons from molecule calculation
            self.results['photons'][i] = np.sum(bkg_subtract)
            self.mean_psf += bkg_subtract/self.results['photons'][i]

        self.results['maxima_px'] = self.positions
        self.results['sharpness'] = self.sharpness
        self.results['roundness'] = self.roundness
        self.results['brightness'] = self.brightness

# FIXME: not saving properly

def fit_area(area, fwhm, bkg):

    # First guess of parameters
#    F = fwhm / (2 * np.sqrt(np.log(2)))
    A = area[np.floor(area.shape[0]/2), np.floor(area.shape[1]/2)] - bkg
    A /= 0.65
    x0, y0 = center_of_mass(area)

    # TODO: get error of each parameter from the fit (see Powell?)
    # TODO: gradient methods not working
#    results = minimize(logll, [A, x0, y0, bkg], args=(fwhm, area),
#                       method='CG', options={'disp': True})
    results = minimize(logll, [A, x0, y0, bkg], args=(fwhm, area),
                       method='Powell')
    return [results.x[0], np.array([results.x[1], results.x[2]]), results.x[3]]


def dexp(x, x0, sigma):
    return np.exp(-((x + 1 - x0)/sigma)**2) - np.exp(-((x - x0)/sigma)**2)


def derf(x, x0, sigma):
    """ Auxiliary  function. x, x0 and sigma are in px units. """
    return erf((x + 1 - x0) / sigma) - erf((x - x0) / sigma)


def lambda_g(A, x0, y0, fwhm, size):
    """ Theoretical mean number of photons detected in an area of size size**2
    due to the emission of a molecule located in (x0, y0). The model PSF is
    a 2d symmetric gaussian of A amplitude with full-width half maximum fwhm.
    x, x0 and fwhm are in px units.
    """
#    fwhm *= 0.5*(np.log(2))**(-1/2)
    fwhm *= 0.6

    derfx = derf(np.arange(size), x0, fwhm)
    derfy = derf(np.arange(size), y0, fwhm)
    return 0.25 * A * fwhm**2 * np.pi * derfx[:, np.newaxis] * derfy


def logll(parameters, *args):
    """ Log-likelihood function for an area of size size**2 around a local
    maximum with respect with a 2d symmetric gaussian of A amplitude centered
    in (x0, y0) with full-width half maximum fwhm on top of a background bkg
    as the model PSF. x, x0 and sigma are in px units.
    """
    A, x0, y0, bkg = parameters
    fwhm, area = args

    lambda_p = lambda_g(A, x0, y0, fwhm, area.shape[0]) + bkg
    return -np.sum(area * np.log(lambda_p) - lambda_p)


# TODO: Not working, check gradient
def ll_jac(parameters, *args):
    """ Jacobian of the log-likelihood function for an area of size size**2
    around a local maximum with respect with a 2d symmetric gaussian of A
    amplitude centered in (x0, y0) with full-width half maximum fwhm on top of
    a background bkg as the model PSF. x, x0 and sigma are in px units.
    Order of derivatives: A, x0, y0, bkg.
    """
    A, x0, y0, bkg = parameters
    fwhm, area = args

#    fwhm *= 0.5*(np.log(2))**(-1/2)
    size = area.shape[0]
    x, y = np.arange(size), np.arange(size)

    derfx = derf(x, x0, fwhm*0.6)
    derfy = derf(y, y0, fwhm*0.6)
    factor = 1 - area/(lambda_g(A, x0, y0, fwhm, size) + bkg)

    jac = np.zeros(4)
    # dL/d(A)
    # The derivative of lambda_g is lambda_g(A=1)
    jac[0] = -np.sum(factor*lambda_g(1, x0, y0, fwhm, size))

    # dL/d(x0) y dL/d(y0)
    # 0.3 = 0.5*0.6
    jac12 = 0.3*A*fwhm*np.sqrt(np.pi)
    jac[1] = jac12*np.sum(dexp(x, x0, fwhm*0.6)[:, np.newaxis]*derfy*factor)
    jac[2] = jac12*np.sum(dexp(y, y0, fwhm*0.6)[:, np.newaxis]*derfx*factor)

    # dL/d(bkg)
    jac[3] = -np.sum(factor)

    return jac


def ll_hess(params, *args):

    A, x0, y0, bkg = params
    pico, F = args

    x, y = np.arange(pico.shape[0]), np.arange(pico.shape[1])

    erfi = derf(x, x0, F)
    erfj = derf(y, y0, F)
    erfij = erfi*erfj
    expi = ex(x, x0 - 1, F) - ex(x, x0, F)
    expj = ex(y, y0 - 1, F) - ex(y, y0, F)

    hess = np.zeros((4, 4))

    # All derivatives made with sympy

    # expr.diff(A, A)
    hess[0, 0] = - np.sum(0.616850275068085 * F**4 * pico * erfi**2 * erfj**2 /
                          ((np.pi/4) * A * F**2 * (erfi * erfj + bkg)**2))

    # expr.diff(A, x0)
    hessi01 = F*expi*erfj*(-1.23370055013617*A*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2 + 1.5707963267949*pico/((np.pi/4)*A*F**2*erfij + bkg) - 1.5707963267949)/np.sqrt(np.pi)
    hess[0, 1] = np.sum(hessi01)
    hess[1, 0] = hess[0, 1]

    # expr.diff(A, y0)
    hess[0, 2] = np.sum(F*expj*(-erfi)*(-1.23370055013617*A*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2 + (np.pi/2)*pico/((np.pi/4)*A*F**2*erfij + bkg) - (np.pi/2))/np.sqrt(np.pi))
    hess[2, 0] = hess[0, 2]

    # expr.diff(A, bkg)
    hessi03 = -(np.pi/4)*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2
    hess[0, 3] = np.sum(hessi03)
    hess[3, 0] = hess[0, 3]

    # expr.diff(x0, x0)
    hess[1, 1] = np.sum(A*erfj*(2.46740110027234*A*F**2*pico*expi**2*(-erfj)/(np.pi*((np.pi/4)*A*F**2*erfij + bkg)**2) - np.pi*pico*((x - x0)*ex(x, x0, F) - (x - x0 + 1)*ex(x, x0 - 1, F))/(np.sqrt(np.pi)*F*((np.pi/4)*A*F**2*erfij + bkg)) + (np.pi*(x - x0)*ex(x, x0, F) - np.pi*(x - x0 + 1)*ex(x, x0 - 1, F))/(np.sqrt(np.pi)*F)))

    # expr.diff(x0, y0)
    hess[1, 2] = np.sum(A*expi*expj*(-2.46740110027234*A*F**2*pico*erfij/((np.pi/4)*A*F**2*erfij + bkg)**2 + np.pi*pico/((np.pi/4)*A*F**2*erfij + bkg) - np.pi)/np.pi)
    hess[2, 1] = hess[1, 2]

    # expr.diff(x0, bkg)
    hess[1, 3] = np.sum(-(np.pi/2)*A*F*pico*expi*(-erfj)/(np.sqrt(np.pi)*((np.pi/4)*A*F**2*erfij + bkg)**2))
    hess[3, 1] = hess[1, 3]

    # expr.diff(y0, y0)
    hess[2, 2] = np.sum(A*(-erfi)*(-2.46740110027234*A*F**2*pico*expj**2*(-erfi)/(np.pi*((np.pi/4)*A*F**2*erfij + bkg)**2) - np.pi*pico*((y - y0)*ex(y, y0, F) - (y - y0 + 1)*ex(y, y0 -1, F))/(np.sqrt(np.pi)*F*((np.pi/4)*A*F**2*erfij + bkg)) + (np.pi*(y - y0)*ex(y, y0, F) - np.pi*(y - y0 + 1)*ex(y, y0 -1, F))/(np.sqrt(np.pi)*F)))

    # expr.diff(y0, bkg)
    hess[2, 3] = np.sum(-(np.pi/2)*A*F*pico*expj*(-erfi)/(np.sqrt(np.pi)*((np.pi/4)*A*F**2*erfij + bkg)**2))
    hess[3, 2] = hess[2, 3]

    # expr.diff(bkg, bkg)
    hess[3, 3] = np.sum(-pico/((np.pi/4)*A*F**2*erfij + bkg)**2)

    return hess

if __name__ == "__main__":

    import tormenta.analysis.stack as stack

    stack = stack.Stack()
    maxima = Maxima(stack.image[10], stack.fwhm)
