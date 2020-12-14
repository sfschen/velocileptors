import numpy as np
import time

from scipy.interpolate import interp1d

from Utils.loginterp import loginterp
from Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from Utils.qfuncfft import QFuncFFT

class KECLEFT:
    '''
    Class to calculate power spectra up to one loop in "expanded LPT,"
    i.e. wherein the long displacement A_{ij} are expanded but the bias basis is Lagrangian.
    
    This is like CLEFT, but with exponent "E"xpanded and separated into powers of k.
    
    All bias tables are formatted as 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1b3 (ncol = 12).
    
    '''

    def __init__(self, k, p, one_loop=True, shear=True, third_order= True, cutoff=20, jn=5, N = 4000, threads=1, extrap_min = -4, extrap_max = 3, import_wisdom=False, wisdom_file='./zelda_wisdom.npy'):

        
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
            self.num_power_components = 12
        elif self.shear:
            self.num_power_components = 10
        else:
            self.num_power_components = 6
        
        self.jn = jn
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

    # The various contributions to P(k) are organized into
    # (1) Linear Theory
    # (2) Connected: this comes from terms that come from connected LPT cumulants that can be Fourier-transformed directly
    # (3) p_k#: this comes from disconnected contributions proportional to k^#
    # Once separated these Hankel transform individually at once for all k.

    def compute_p_linear(self):
        self.p_linear = np.zeros( (self.num_power_components, self.N) )
        self.p_linear[0,:] = self.pint
        self.p_linear[1,:] = 2*self.pint
        self.p_linear[2,:] = self.pint
        

    def compute_p_connected(self):
        
        self.p_connected = np.zeros( (self.num_power_components, self.N) )
        self.p_connected[0,:] = 9./98 * self.qf.Q1 + 10./21 * self.qf.R1 + 3./7 * (2*self.qf.R2 + self.qf.Q2)
        self.p_connected[1,:] = 10./21*self.qf.R1 + 1./7*(6*self.qf.R1 + 12*self.qf.R2 + 6*self.qf.Q5)
        self.p_connected[2,:] = 6./7 * (self.qf.R1 + self.qf.R2)
        self.p_connected[3,:] = 3./7 * self.qf.Q8
        
        if self.shear or self.third_order:
            self.p_connected[6,:] = 2./7*self.qf.Qs2
            
        if self.third_order:
            self.p_connected[10,:] = 2 * self.qf.Rb3 * self.pint
            self.p_connected[11,:] = 2 * self.qf.Rb3 * self.pint
    
    def compute_p_k0(self):
        self.p_k0 = np.zeros( (self.num_power_components, self.N) )
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        for l in range(1):
            mu0fac = (l == 0)
            
            bias_integrands[5,:] = mu0fac * (0.5*self.corlin**2)
                                   
            if self.shear or self.third_order:
                bias_integrands[8,:] = mu0fac * self.chi
                bias_integrands[9,:] = mu0fac * self.zeta
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            self.p_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    
    def compute_p_k1(self):
        self.p_k1 = np.zeros( (self.num_power_components, self.N) )
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        for l in [1]:
            mu1fac = (l == 1)
            
            bias_integrands[4,:] = mu1fac * (-2*self.Ulin*self.corlin)
            
            if self.shear or self.third_order:
                bias_integrands[7,:] = mu1fac * (-2*self.V)
                                   
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            self.p_k1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
            
            
    def compute_p_k2(self):
        self.p_k2 = np.zeros( (self.num_power_components, self.N) )
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        for l in range(3):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            
            bias_integrands[2,:] = mu0fac * (- 0.5*self.Xlin*self.corlin)  + \
                                   mu2fac * (-0.5*self.Ylin*self.corlin - self.Ulin**2)
            bias_integrands[3,:] = mu2fac * (-self.Ulin**2)
            
            if self.shear or self.third_order:
                bias_integrands[6,:] = mu0fac * (-self.Xs2) + mu2fac * (-self.Ys2)
                                   
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            self.p_k2 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_p_k3(self):
        self.p_k3 = np.zeros( (self.num_power_components, self.N) )
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
                
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)

            
            bias_integrands[1,:] = mu1fac * (self.Ulin*self.Xlin) + \
                                   mu3fac * (self.Ulin*self.Ylin)
                                   
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            self.p_k3 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    
    
    def compute_p_k4(self):
        self.p_k4 = np.zeros( (self.num_power_components, self.N) )
        ret = np.zeros(self.num_power_components)
        
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)

            
            bias_integrands[0,:] = mu0fac * (+1./8*self.Xlin**2 ) + \
                                   mu2fac * (1./4*self.Xlin*self.Ylin  ) + \
                                   mu4fac * (1./8*self.Ylin**2)
                                   
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            self.p_k4 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def make_ptable(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''
        Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        for ii in range(self.num_power_components):
            self.pktable[:, ii+1] = interp1d(self.kint,\
                                       self.p_linear[ii,:] + self.p_connected[ii,:] \
                                       + self.p_k0[ii,:] + self.kint * self.p_k1[ii,:] + self.kint**2 * self.p_k2[ii,:]\
                                       + self.kint**3 * self.p_k3[ii,:] + self.kint**4 * self.p_k4[ii,:])(kv)


    def export_wisdom(self, wisdom_file='./wisdom.npy'):
        self.sph.export_wisdom(wisdom_file=wisdom_file)
