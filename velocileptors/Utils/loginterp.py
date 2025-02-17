import numpy as np
from scipy.interpolate import make_interp_spline

def loginterp(x, y):
    '''
    Extrapolate function by evaluating a log-index of left & right side.
    
    From Chirag Modi's CLEFT code at
    https://github.com/modichirag/CLEFT/blob/master/qfuncpool.py
    
    Updated since scipy no longer supports "derivative" and wants us to use a new spline object (???).
    '''

    # Find left most point with two consecutive points of the same sign, and same for right
    lp = 0
    while np.sign(y[lp]) != np.sign(y[lp+1]):
        if lp < len(x) - 2:
            lp += 1
        else:
            raise Exception("The input points never stop changing sign!")
    
    rp = len(x) - 1
    while np.sign(y[rp]) != np.sign(y[rp-1]):
        if rp > 2:
            rp -= 1
        else:
            raise Exception("The input points never stop changing sign!")
            
    #print(lp, rp)

    # This spline forces the second log derivative at the boundaries to be zero
    yint = make_interp_spline(np.log(x[lp:(rp+1)]), y[lp:(rp+1)], bc_type='natural')
            
    # Now compute the slope at the stable points
    deriv = yint.derivative()

    lneff, rneff = deriv(np.log(x[lp]))/y[lp], deriv(np.log(x[rp]))/y[rp]
    #print(lneff, rneff)
    
    # nan_to_numb is to prevent (xx/x[l/r])^lneff to go to nan on the other side
    # since this value should be zero on the wrong side anyway

    yint2 = lambda xx:  (xx <= x[lp]) * y[lp]* np.nan_to_num((xx/x[lp])**lneff) \
                   + (xx >= x[rp]) * y[rp]* np.nan_to_num((xx/x[rp])**rneff) \
                   + (xx > x[lp]) * (xx < x[rp]) * yint(np.log(xx))

    return yint2
