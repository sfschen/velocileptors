import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as Spline

def gaussian_poly_extrap(kout, kint, pint, frac = 1):
    '''
    Extrapolates beyond the end of kint by a damped polynomial in kint (i.e. Hermite).
    Does nothing on the low k end.
    
    The extrapolation form is
    (A + B k) * exp(- k^2/k0^2) where k0 is taken to be some fraction (1) of the
    final element of kint.
    
    '''
    
    # Solve for the coefficients
    k1, k2 = kint[-2], kint[-1]
    p1, p2 = pint[-2], pint[-1]
    k0 = frac * k2
    
    B = (p2 * np.exp(k2**2/k0**2) - p1 * np.exp(k1**2/k0**2)) / (k2 - k1)
    A = p2 * np.exp(k2**2/k0**2) - B * k2
    
    # Interpolate/extrapolate
    ret = np.zeros_like(kout)
    extrap_iis = (kout > k2)
    ret[~extrap_iis] = Spline(kint,pint)(kout[~extrap_iis])
    ret[extrap_iis] = ( (A + B*kout) * np.exp(-kout**2/k0**2) )[extrap_iis]
    
    return ret
    
    
    
    
    
    