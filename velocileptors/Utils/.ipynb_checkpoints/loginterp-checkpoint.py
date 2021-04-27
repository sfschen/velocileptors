import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import inspect

def loginterp(x, y, yint = None, side = "both", lorder = 9, rorder = 9, lp = 1, rp = -2,
              ldx = 1e-6, rdx = 1e-6,\
              interp_min = -12, interp_max = 12, Nint = 10**5, verbose=False, option='B'):
    '''
    Extrapolate function by evaluating a log-index of left & right side.
    
    From Chirag Modi's CLEFT code at
    https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
    
    The warning for divergent power laws on both ends is turned off. To turn back on uncomment lines 26-33.
    '''
    
    if yint is None:
        yint = interpolate(x, y, k = 5)
    if side == "both":
        side = "lr"
    
    # Make sure there is no zero crossing between the edge points
    # If so assume there can't be another crossing nearby
    
    if np.sign(y[lp]) == np.sign(y[lp-1]) and np.sign(y[lp]) == np.sign(y[lp+1]):
        l = lp
    else:
        l = lp + 2
        
    if np.sign(y[rp]) == np.sign(y[rp-1]) and np.sign(y[rp]) == np.sign(y[rp+1]):
        r = rp
    else:
        r = rp - 2
    
    lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
    rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
    
    #print(lneff, rneff)
    
    # uncomment if you like warnings.
    #if verbose:
    #    if lneff < 0:
    #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
    #        print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
    #    if rneff > 0:
    #        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
    #        print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

    if option == 'A':
    
        xl = np.logspace(interp_min, np.log10(x[l]), Nint)
        xr = np.logspace(np.log10(x[r]), interp_max, Nint)
        yl = y[l]*(xl/x[l])**lneff
        yr = y[r]*(xr/x[r])**rneff
        #print(xr/x[r])

        xint = x[l+1:r].copy()
        yint = y[l+1:r].copy()
        if side.find("l") > -1:
            xint = np.concatenate((xl, xint))
            yint = np.concatenate((yl, yint))
        if side.find("r") > -1:
            xint = np.concatenate((xint, xr))
            yint = np.concatenate((yint, yr))
        yint2 = interpolate(xint, yint, k = 5, ext=3)
    
    else:
        yint2 = lambda xx: (xx <= x[l]) * y[l]*(xx/x[l])**lneff \
                         + (xx >= x[r]) * y[r]*(xx/x[r])**rneff \
                         + (xx > x[l]) * (xx < x[r]) * interpolate(x, y, k = 5, ext=3)(xx)

    return yint2
