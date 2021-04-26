import numpy as np

import os

from scipy.interpolate import interp1d

from velocileptors.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from velocileptors.Utils.qfuncfft import QFuncFFT
from velocileptors.Utils.loginterp import loginterp

class CLEFT:
    '''
    Class to calculate power spectra up to one loop.
    
    Based on Chirag's code
    
    https://github.com/sfschen/velocileptors/blob/master/LPT/cleft_fftw.py
    
    The bias parameters are ordered in pktable as
    1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
    where b3 is a catch-all for third order bias parameters degenerate at one-loop order.
    
    Can combine into a full one-loop real-space power spectrum using the function combine_bias_terms_pk.
    
    '''

    def __init__(self, k, p, one_loop=True, shear=True, third_order=True, cutoff=10, jn=5, N = 2000, threads=None, extrap_min = -5, extrap_max = 3, import_wisdom=False, wisdom_file='wisdom.npy'):

        
        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        self.one_loop = one_loop
        self.shear = shear
        self.third_order = third_order
        
        self.update_power_spectrum(k,p)
        
        self.pktable = None
        if self.third_order:
            self.num_power_components = 13
        elif self.shear:
            self.num_power_components = 11
        else:
            self.num_power_components = 7
        
        self.jn = jn
        
        if threads is None:
            self.threads = int( os.getenv("OMP_NUM_THREADS","1") )
        else:
            self.threads = threads
    
        self.import_wisdom = import_wisdom
        self.wisdom_file = wisdom_file
        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        

    def update_power_spectrum(self, k, p):
        # Updates the power spectrum and various q functions. Can continually compute for new cosmologies without reloading FFTW
        self.k = k
        self.p = p
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()

    def setup_powerspectrum(self):
        
        # This sets up terms up to one looop in the combination (symmetry factors) they appear in pk
        
        self.qf = QFuncFFT(self.kint, self.pint, qv=self.qint, oneloop=self.one_loop, shear=self.shear, third_order=self.third_order)
        
        # linear terms
        self.Xlin = self.qf.Xlin
        self.Ylin = self.qf.Ylin
        
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        
        self.Ulin = self.qf.Ulin
        self.corlin = self.qf.corlin
    
        if self.one_loop:
        # one loop terms: here we add in all the symmetry factors
            self.Xloop = 2 * self.qf.Xloop13 + self.qf.Xloop22; self.sigmaloop = self.Xloop[-1]
            self.Yloop = 2 * self.qf.Yloop13 + self.qf.Yloop22
    
            self.Vloop = 3 * (2 * self.qf.V1loop112 + self.qf.V3loop112) # this multiplies mu in the pk integral
            self.Tloop     = 3 * self.qf.Tloop112 # and this multiplies mu^3
    
            self.X10 = 2 * self.qf.X10loop12
            self.Y10 = 2 * self.qf.Y10loop12
            self.sigma10 = (self.X10 + self.Y10)[-1]
    
            self.U3 = self.qf.U3
            self.U11 = self.qf.U11
            self.U20 = self.qf.U20
            self.Us2 = self.qf.Us2

        else:
            self.Xloop, self.Yloop, self.sigmaloop, self.Vloop, self.Tloop, self.X10, self.Y10, self.sigma10, self.U3, self.U11, self.U20, self.Us2 = (0,)*12

        # load shear functions
        if self.shear or self.third_order:
            self.Xs2 = self.qf.Xs2
            self.Ys2 = self.qf.Ys2; self.sigmas2 = (self.Xs2 + self.Ys2)[-1]
            self.V = self.qf.V
            self.zeta = self.qf.zeta
            self.chi = self.qf.chi
            
        if self.third_order:
            self.Ub3 = self.qf.Ub3
            self.theta = self.qf.theta

    def p_integrals(self, k):
        '''
        Compute P(k) for a single k as a vector of all bias contributions.
        
        '''
        ksq = k**2; kcu = k**3; k4 = k**4
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5 * ksq *self.sigma)
        
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            shiftfac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = 1. - 2.*(l-1)/ksq/self.Ylin # mu3 terms start at j1 so l -> l-1
            mu4fac = 1 - 4*l/ksq/self.Ylin + 4*l*(l-1)/(ksq*self.Ylin)**2
            
            bias_integrands[0,:] = 1. - 0.5 * ksq * (self.Xloop + mu2fac * self.Yloop ) + kcu * shiftfac * (self.Vloop + self.Tloop * mu3fac)/6. # matter
            bias_integrands[1,:] = (-2 * k * (self.Ulin+self.U3)) * shiftfac - ksq*(self.X10 + self.Y10*mu2fac) # b1
            bias_integrands[2,:] = self.corlin - ksq*mu2fac*self.Ulin**2 - shiftfac*k*self.U11 # b1sq
            bias_integrands[3,:] = - ksq * mu2fac * self.Ulin**2 - shiftfac*k*self.U20 # b2
            bias_integrands[4,:] = (-2 * k * self.Ulin * self.corlin) * shiftfac # b1b2
            bias_integrands[5,:] = 0.5 * self.corlin**2 # b2sq
            
            if self.shear or self.third_order:
                bias_integrands[6,:] = -ksq * (self.Xs2 + mu2fac*self.Ys2) - 2*k*self.Us2*shiftfac # bs should be both minus
                bias_integrands[7,:] = -2*k*self.V*shiftfac # b1bs
                bias_integrands[8,:] = self.chi # b2bs
                bias_integrands[9,:] = self.zeta # bssq
                
            if self.third_order:
                bias_integrands[10,:] = -2 * k * self.Ub3 * shiftfac #bs
                bias_integrands[11,:] = 2 * self.theta #b1 bs
                
            bias_integrands[-1,:] = 1 # this is the counterterm, minus a factor of k2


            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
    
    
        #ret += ret[0] * zero_lags
        
        return 4*suppress*np.pi*ret

    def make_ptable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo])

    def combine_bias_terms_pk(self, b1, b2, bs, b3, alpha, sn):
        '''
        Combine all the bias terms into one power spectrum,
        where alpha is the counterterm and sn the shot noise/stochastic contribution.
        
        Three options, for
        
        (1) Full one-loop bias expansion (third order bias)
        (2) only quadratic bias, including shear
        (3) only density bias
        
        If (2) or (3), i.e. the class is set such that shear=False or third_order=False then the bs
        and b3 parameters are not used.
        
        '''
        arr = self.pktable
        
        
        if self.third_order:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        elif self.shear:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
        else:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2])
            

        kv = arr[:,0]; za = arr[:,-1]
        pktemp = np.copy(arr)[:,1:-1]

        res = np.sum(pktemp * bias_monomials, axis =1) + alpha*kv**2 * za + sn

        return kv, res


    def export_wisdom(self, wisdom_file='./wisdom.npy'):
        self.sph.export_wisdom(wisdom_file=wisdom_file)
