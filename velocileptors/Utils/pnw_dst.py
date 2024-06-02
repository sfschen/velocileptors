import numpy as np
from loginterp import loginterp

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelmin, argrelmax, tukey
from scipy.fftpack import dst, idst
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

def pnw_dst(k,p, ii_l=None,ii_r=None,extrap_min=1e-3, extrap_max=10, N=16,verbose=False):
    '''
    Implement the wiggle/no-wiggle split procedure from Benjamin
    Wallisch's thesis (arXiv:1810.02800)
    '''

    # put wavenumbers onto a linear grid
    ks = np.linspace( extrap_min, extrap_max, 2**N)
    lnps = Spline(k, np.log(k*p), ext=1)(ks)
 
    
    # sine transform
    dst_ps = dst(lnps)
    dst_odd = dst_ps[1::2]
    dst_even = dst_ps[0::2]
    
    # find the BAO regions
    if ii_l is None or ii_r is None:
        d2_even = np.gradient( np.gradient(dst_even) )
        ii_l = argrelmin(gaussian_filter(d2_even,4))[0][0]
        ii_r = argrelmax(gaussian_filter(d2_even,4))[0][1]
        if verbose: print(ii_l,ii_r)
    
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis); iis_div[0] = 1.
        #cutiis_odd = (iis > (ii_l-3) ) * (iis < (ii_r+20) )
        cutiis_even = (iis > (ii_l-3) ) *  (iis < (ii_r+10) )
        
        d2_odd = np.gradient( np.gradient(dst_odd) )
        ii_l = argrelmin(gaussian_filter(d2_odd,4))[0][0]
        ii_r = argrelmax(gaussian_filter(d2_odd,4))[0][1]
        if verbose: print(ii_l,ii_r)
    
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis); iis_div[0] = 1.
        cutiis_odd = (iis > (ii_l-3) ) * (iis < (ii_r+20) )
        #cutiis_even = (iis > (ii_l-3) ) *  (iis < (ii_r+10) )
        
    else:
        iis = np.arange(len(dst_odd))
        iis_div = np.copy(iis); iis_div[0] = 1.
        cutiis_odd = (iis > (ii_l) ) * (iis < (ii_r) )
        cutiis_even = (iis > (ii_l) ) *  (iis < (ii_r) )

    # ... and interpolate over them
    interp_odd = interp1d(iis[~cutiis_odd],(iis**2*dst_odd)[~cutiis_odd],kind='cubic')(iis)/iis_div**2 
    interp_odd[0] = dst_odd[0]
    
    interp_even = interp1d(iis[~cutiis_even],(iis**2*dst_even)[~cutiis_even],kind='cubic')(iis)/iis_div**2 
    interp_even[0] = dst_even[0]
    
    # Transform back
    interp = np.zeros_like(dst_ps)
    interp[0::2] = interp_even
    interp[1::2] = interp_odd

    lnps_nw = idst(interp) / 2**17
    
    return k, Spline(ks, np.exp(lnps_nw)/ks,ext=1)(k)
