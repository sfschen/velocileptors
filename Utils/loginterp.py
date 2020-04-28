import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import inspect

def loginterp(x, y, yint = None, side = "both", lorder = 9, rorder = 9, lp = 1, rp = -1,
              ldx = 1e-6, rdx = 1e-6):
    '''
    Extrapolate function by evaluating a log-index of left & right side.
    
    From Chirag Modi's CLEFT code at
    https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
    '''
    
    if yint is None:
        yint = interpolate(x, y, k = 5)
    if side == "both":
        side = "lr"
    l =lp
    r =rp
    lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
    rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
    if lneff < 0:
        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        print('WARNING: Runaway index on left side, bad interpolation. Left index = %0.3e at %0.3e'%(lneff, x[l]))
    if rneff > 0:
        print( 'In function - ', inspect.getouterframes( inspect.currentframe() )[2][3])
        print('WARNING: Runaway index on right side, bad interpolation. Reft index = %0.3e at %0.3e'%(rneff, x[r]))

    xl = np.logspace(-12, np.log10(x[l]), 10**5)
    xr = np.logspace(np.log10(x[r]), 12., 10**5)
    yl = y[l]*(xl/x[l])**lneff
    yr = y[r]*(xr/x[r])**rneff

    xint = x[l+1:r].copy()
    yint = y[l+1:r].copy()
    if side.find("l") > -1:
        xint = np.concatenate((xl, xint))
        yint = np.concatenate((yl, yint))
    if side.find("r") > -1:
        xint = np.concatenate((xint, xr))
        yint = np.concatenate((yint, yr))
    yint2 = interpolate(xint, yint, k = 5, ext=3)

    return yint2
