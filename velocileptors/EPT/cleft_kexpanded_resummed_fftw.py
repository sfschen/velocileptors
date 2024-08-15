import numpy as np

import time

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from velocileptors.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from velocileptors.Utils.pnw_dst import pnw_dst

from velocileptors.EPT.cleft_kexpanded_fftw import KECLEFT

class RKECLEFT:
    '''
    Class based on cleft_kexpanded_fftw to IR resummed real space spectra, in expanded LPT.

    
    '''

    def __init__(self, k, p, pnw=None, N=4000, *args, rbao = 110, sbao=None, **kw):
        
        
        self.rbao = rbao
        
        # construct pair of wiggle/no-wiggle objects
        self.cleft = KECLEFT( k, p, third_order=True, N=N, **kw)
        self.cleft.compute_p_linear()
        self.cleft.compute_p_connected()
        self.cleft.compute_p_k0()
        self.cleft.compute_p_k1()
        self.cleft.compute_p_k2()
        self.cleft.compute_p_k3()
        self.cleft.compute_p_k4()
        
        if pnw is None:
            # savgol was not reliable at high k
            #knw = self.cleft.kint
            #Nfilter =  np.ceil(np.log(7) /  np.log(knw[-1]/knw[-2])) // 2 * 2 + 1 # filter length ~ log span of one oscillation from k = 0.01
            #pnw = savgol_filter(self.cleft.pint, int(Nfilter), 4)
            # use pnw_dst from Ben Wallisch's thesis
            knw, pnw = pnw_dst(k, p)
        else:
            knw, pnw = k, pnw
            
        self.cleft_nw = KECLEFT( knw, pnw, third_order=True, N=N, **kw)
        self.cleft_nw.compute_p_linear()
        self.cleft_nw.compute_p_connected()
        self.cleft_nw.compute_p_k0()
        self.cleft_nw.compute_p_k1()
        self.cleft_nw.compute_p_k2()
        self.cleft_nw.compute_p_k3()
        self.cleft_nw.compute_p_k4()
                
        # compute BAO damping
        if sbao is None:
            self.sigma_squared_bao = np.interp(self.rbao,\
                                               self.cleft.qint, self.cleft.Xlin + self.cleft.Ylin)
        else:
            self.sigma_squared_bao = sbao

        
        
    def make_ptable(self, D = 1, kmin = 1e-3, kmax = 3, nk = 100):
        
        self.kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.damp_exp = - 0.5 * self.kv**2 * D**2 * self.sigma_squared_bao
        self.damp_fac = np.exp(self.damp_exp)
        
        self.pnwtable_linear = self.cleft_nw.make_ptable(D = D, nonlinear=0, kmin = kmin, kmax = kmax, nk = nk)
        self.pwtable_linear = self.cleft.make_ptable(D = D, nonlinear=0, kmin = kmin, kmax = kmax, nk = nk)
        
        self.pnwtable = self.cleft_nw.make_ptable(D = D, nonlinear=1, kmin = kmin, kmax = kmax, nk = nk)
        self.pwtable = self.cleft.make_ptable(D = D, nonlinear=1, kmin = kmin, kmax = kmax, nk = nk)
        
        self.pktable = self.pnwtable + self.damp_fac[:,None] * (self.pwtable - self.pnwtable)\
                                     - (self.damp_exp * self.damp_fac)[:,None] * (self.pwtable_linear - self.pnwtable_linear)
        
        
    
