import numpy as np


import time

from scipy.interpolate import interp1d

from Utils.loginterp import loginterp
from Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from Utils.qfuncfft import QFuncFFT

from LPT.cleft_fftw import CLEFT

class VelocityMoments(CLEFT):
    '''
    Class based on cleft_fftw to compute pairwise velocity moments.
    '''

    def __init__(self, *args, beyond_gauss = False, **kw):
        '''
           Same keywords as the cleft_fftw class. Go look there!
        '''
        
        # Set up the configuration space quantities
        CLEFT.__init__(self, *args, **kw)
        
        self.beyond_gauss = beyond_gauss
        
        self.setup_onedot()
        self.setup_twodots()
        
        # v12 and sigma12 only have a subset of the bias contributions so we don't need to have as many FFTs
        if self.third_order:
            self.num_vel_components = 8; self.vii = np.array([0,1,2,3,4,6,7,10]) + 1
            self.num_spar_components = 5; self.sparii = np.array([0,1,2,3,6]) + 1
            self.num_strace_components = 5; self.straceii = np.array([0,1,2,3,6]) + 1
        elif self.shear:
            self.num_vel_components = 7; self.vii = np.array([0,1,2,3,4,6,7]) + 1
            self.num_spar_components = 5; self.sparii = np.array([0,1,2,3,6]) + 1
            self.num_strace_components = 5; self.straceii = np.array([0,1,2,3,6]) + 1
        else:
            self.num_vel_components = 5; self.vii = np.array([0,1,2,3,4]) + 1
            self.num_spar_components = 4; self.sparii = np.array([0,1,2,3]) + 1
            self.num_strace_components = 4; self.straceii = np.array([0,1,2,3]) + 1
                
        # Need one extra component to do the matter za
        self.sph_v = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_vel_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.sph_spar = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_spar_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        self.sph_strace = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_strace_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
        
        
        if self.beyond_gauss:
            # Beyond the first two moments
            self.num_gamma_components = 2; self.gii = np.array([0,1]) + 1 # gamma has matter (all loop, so lump into 0) and b1
            self.sph_gamma1 = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_gamma_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)
            self.sph_gamma2 = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_gamma_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)

            # fourth moment
            self.num_kappa_components = 3; self.kii = np.array([0,1,2]) + 1 # note that these are not the bias comps
            self.sph_kappa = SphericalBesselTransform(self.qint, L=self.jn, ncol=(self.num_kappa_components), threads=self.threads, import_wisdom= self.import_wisdom, wisdom_file = self.wisdom_file)


        

    def update_power_spectrum(self,k,p):
        '''
        Same as the one in cleft_fftw but also do the velocities.
        '''
        super(VelocityMoments,self).update_power_spectrum(k,p)
        self.setup_onedot()
        self.setup_twodots()
        self.setup_threedots()

        

    def setup_onedot(self):
        '''
        Create quantities linear in f. All quantities are with f = 1, since converting back is trivial.
        '''
        self.Xdot = self.Xlin; self.sigmadot = self.Xdot[-1]
        self.Ydot = self.Ylin
        
        self.Vdot = 4./3 * self.Vloop # these are only the symmetrized version since all we need...
        self.Tdot = 4./3 * self.Tloop # is k_i k_j k_k W_{ijk}

        self.Udot = self.Ulin
        self.Uloopdot  = 3 * self.U3

        self.U11dot = 2 * self.U11
        self.U20dot = 2 * self.U20
    
        # some one loop terms have to be explicitly set to zero
        if self.one_loop:
            self.Xloopdot  = (4 * self.qf.Xloop13 + 2 * self.qf.Xloop22) * self.one_loop; self.sigmaloopdot = self.Xloopdot[-1]
            self.Yloopdot  = (4 * self.qf.Yloop13 + 2 * self.qf.Yloop22) * self.one_loop
            self.X10dot = 1.5 * self.X10; self.sigma10dot = self.X10dot[-1]
            self.Y10dot = 1.5 * self.Y10
        else:
            self.Xloopdot = 0; self.sigmaloopdot = 0
            self.Yloopdot = 0
            self.X10dot = 0; self.sigma10dot = 0
            self.Y10dot = 0
        
        if self.shear:
            self.Us2dot = 2 * self.Us2
            self.V12dot = self.V
            self.Xs2dot = self.Xs2; self.sigmas2dot = self.Xs2dot[-1]
            self.Ys2dot = self.Ys2
            
        if self.third_order:
            self.Ub3dot = self.Ub3

    def setup_twodots(self):
        '''
        Same as onedot but now for those quadratic in f.
        '''
        self.Xddot = self.Xlin; self.sigmaddot = self.Xddot[-1]
        self.Yddot = self.Ylin
        
        # Here we will need two forms, one symmetrized:
        self.Vddot = 5./3 * self.Vloop #these are only the symmetrized version since all we need...
        self.Tddot = 5./3 * self.Tloop # is k_i k_j k_k W_{ijk}

        # Explicitly set certain terms to zero if not one loop
        if self.one_loop:
            self.Xloopddot = (4 * self.qf.Xloop22 + 6 * self.qf.Xloop13) * self.one_loop; self.sigmaloopddot = self.Xloopddot[-1]
            self.Yloopddot = (4 * self.qf.Yloop22 + 6 * self.qf.Yloop13) * self.one_loop
        
            self.X10ddot = 2 * self.X10; self.sigma10ddot = self.X10ddot[-1]
            self.Y10ddot = 2 * self.Y10
        
            # and the other from k_i \delta_{jk} \ddot{W}_{ijk}
            self.kdelta_Wddot = (18 * self.qf.V1loop112 + 7 * self.qf.V3loop112 + 5 * self.qf.Tloop112) * self.one_loop
        else:
            self.Xloopddot = 0; self.sigmaloopddot = 0
            self.Yloopddot = 0
            self.X10ddot = 0; self.sigma10ddot = 0
            self.Y10ddot = 0
            self.kdelta_Wddot = 0
        
        if self.shear:
            self.Xs2ddot = self.Xs2; self.sigmas2ddot = self.Xs2ddot[-1]
            self.Ys2ddot = self.Ys2

    def setup_threedots(self):
        self.Vdddot = 2 * self.Vloop
        self.Tdddot = 2 * self.Tloop

    def v_integrals(self,k):

        '''
        Gives bias contributions to v(k) at a given k.
        '''
        
        ksq = k**2; kcu = k**3
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_vel_components)
        
        bias_integrands = np.zeros( (self.num_vel_components,self.N)  )
                
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = mu1fac * (1. - 2.*(l-1)/ksq/self.Ylin) # mu3 terms start at j1 so l -> l-1
            
            bias_integrands[0,:] = k * (self.Xdot + self.Xloopdot + mu2fac * (self.Ydot + self.Yloopdot)) - 0.5 * ksq * (mu1fac * self.Vdot + mu3fac * self.Tdot) # matter
            bias_integrands[1,:] = 2* (-ksq * self.Ulin * (mu1fac*self.Xdot + mu3fac * self.Ydot) + mu1fac * (self.Udot + self.Uloopdot) + k * (self.X10dot + mu2fac * self.Y10dot) ) # b1
            bias_integrands[2,:] = 2*k*mu2fac*self.Ulin*self.Udot + mu1fac*self.U11dot + k * self.corlin * (self.Xdot + self.Ydot * mu2fac) # b1sq
            bias_integrands[3,:] = 2*k*self.Ulin*self.Udot * mu2fac + self.U20dot * mu1fac # b2
            bias_integrands[4,:] = 2 * self.corlin * self.Udot * mu1fac# b1b2
            
            if self.shear or self.third_order:
                bias_integrands[5,:] = 2*self.Us2dot*mu1fac + 2*k * (self.Xs2dot + mu2fac * self.Ys2dot) #bs: the second factor used to miss a factor of two
                bias_integrands[6,:] = 2*self.V12dot*mu1fac #b1 bs
            if self.third_order:
                bias_integrands[7,:] = 2*self.Ub3dot * mu1fac
            
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_v.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)

        
        return 4*suppress*np.pi*ret
    
    
    def spar_integrals(self,k):
        '''
        Gives bias contributions to \sigma_\parallel at a given k.
        '''
        ksq = k**2; kcu = k**3
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_spar_components)
        
        bias_integrands = np.zeros( (self.num_spar_components,self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = mu1fac * (1. - 2.*(l-1)/ksq/self.Ylin) # mu3 terms start at j1 so l -> l-1
            mu4fac = 1 - 4*l/ksq/self.Ylin + 4*l*(l-1)/(ksq*self.Ylin)**2
            
            bias_integrands[0,:] = self.Xddot + self.Yddot * mu2fac + self.Xloopddot - ksq*self.Xdot**2 + (self.Yloopddot - 2*ksq*self.Xdot*self.Ydot)*mu2fac - ksq*self.Ydot**2*mu4fac - k * (mu1fac * self.Vddot + mu3fac * self.Tddot)  # matter
            bias_integrands[1,:] = 2 * ( self.X10ddot -k*(self.Ulin*self.Xddot + 2*self.Udot*self.Xdot)*mu1fac + self.Y10ddot*mu2fac - k*(self.Ulin*self.Yddot + 2*self.Udot*self.Ydot)*mu3fac ) # b1
            bias_integrands[2,:] = self.corlin*self.Xddot + (self.corlin*self.Yddot + 2*self.Udot**2)*mu2fac # b1sq
            bias_integrands[3,:] = 2 * self.Udot**2 * mu2fac # b2
            
            if self.shear or self.third_order:
                bias_integrands[4,:] = 2 * (self.Xs2ddot + self.Ys2ddot * mu2fac) # bs
            
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_spar.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)

        
        return 4*suppress*np.pi*ret
    
    
    def strace_integrals(self,k):
        '''
            Gives bias contributions to \sigma_\parallel at a given k.
            '''
        ksq = k**2; kcu = k**3
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_strace_components)
        
        bias_integrands = np.zeros( (self.num_strace_components,self.N)  )
                
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = mu1fac * (1. - 2.*(l-1)/ksq/self.Ylin) # mu3 terms start at j1 so l -> l-1
            
            bias_integrands[0,:] = (3 * self.Xddot + self.Yddot) + 3 * self.Xloopddot + self.Yloopddot - ksq*self.Xdot**2 - ksq*(self.Ydot**2+2*self.Xdot*self.Ydot)*mu2fac - k * self.kdelta_Wddot * mu1fac # za
            bias_integrands[1,:] = 2 * ( (3*self.X10ddot + self.Y10ddot) - k*self.Ulin*(3*self.Xddot+self.Yddot)*mu1fac - 2*k*self.Udot*(self.Xdot+self.Ydot)*mu1fac ) # b1
            bias_integrands[2,:] = self.corlin*(3*self.Xddot + self.Yddot) + 2*self.Udot**2 # b1sq
            bias_integrands[3,:] = 2 * self.Udot**2 # b2
            
            if self.shear or self.third_order:
                bias_integrands[4,:] = 2 * (3*self.Xs2ddot + self.Ys2ddot)
            
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_strace.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)

        
        return 4*suppress*np.pi*ret

    
    def gamma1_integrals(self,k):
        '''
        Gives bias contributions to Im[\hk_i \hk_j \hk_k \gamma_{ijk}]
        '''
        ksq = k**2; kcu = k**3
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_gamma_components)
        
        bias_integrands = np.zeros( (self.num_gamma_components,self.N)  )
        
        #zero_lags = np.array([self.sigmadot*self.sigmaddot])
        
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu3fac = mu1fac * (1. - 2.*(l-1)/ksq/self.Ylin) # mu3 terms start at j1 so l -> l-1
            mu4fac = 1 - 4*l/ksq/self.Ylin + 4*l*(l-1)/(ksq*self.Ylin)**2
            
            bias_integrands[0,:] = self.Vdddot*mu1fac+self.Tdddot*mu3fac + 3*k*(self.Xdot*self.Xddot + (self.Xdot*self.Yddot+self.Ydot*self.Xddot)*mu2fac + self.Ydot*self.Yddot * mu4fac ) # matter
            bias_integrands[1,:] = (6*self.Udot*(self.Xddot*mu1fac + self.Yddot*mu3fac))  # b1

            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gamma1.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
        
        return 4*suppress*np.pi*ret

    def gamma2_integrals(self,k):
        '''
            Gives bias contributions to Im[ \hk_i \delta_{jk} \gamma_{ijk} ]
            '''
        ksq = k**2; kcu = k**3
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_gamma_components)
        
        bias_integrands = np.zeros( (self.num_gamma_components,self.N)  )
        
        #zero_lags = np.array([5*self.sigmadot*self.sigmaddot])
        
        for l in range(self.jn):
            # l-dep functions
            mu1fac = (l>0)/(k * self.yq)
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            
            bias_integrands[0,:] = (5*self.Vdddot/3+self.Tdddot)*mu1fac + k*(  5*self.Xdot*self.Xddot+self.Xdot*self.Yddot +(2*self.Xdot*self.Yddot+self.Ydot*(5*self.Xddot+3*self.Yddot) )*mu2fac ) # matter
            bias_integrands[1,:] = (2*self.Udot*(5*self.Xddot + 3*self.Yddot)*mu1fac)  # b1
            
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gamma2.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
    
        return 4*suppress*np.pi*ret
    
    
    def kappa_integrals(self,k):
        '''
        Since kappa_ijk only involes one term we can just do them all in one go.
        
        '''
        ksq = k**2; kf = k**4
        expon = np.exp(-0.5*ksq * (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*ksq *self.sigma)
        
        ret = np.zeros(self.num_kappa_components)
        
        bias_integrands = np.zeros( (self.num_kappa_components,self.N)  )
        
        for l in range(self.jn):
            # l-dep functions
            mu2fac = 1. - 2.*l/ksq/self.Ylin
            mu4fac = 1 - 4*l/ksq/self.Ylin + 4*l*(l-1)/(ksq*self.Ylin)**2

            bias_integrands[0,:] = 15 * self.Xddot**2 + 10 * self.Xddot*self.Yddot + 3 * self.Yddot**2
            bias_integrands[1,:] = 5 * self.Xddot**2 + self.Xddot*self.Yddot + (7*self.Xddot*self.Yddot + 3*self.Yddot**2)*mu2fac
            bias_integrands[2,:] = 3 * self.Xddot**2 + 6*self.Xddot*self.Yddot*mu2fac + 3*self.Yddot**2*mu4fac
            
            if l == 0:
                bias_integrands = bias_integrands * expon
                bias_integrands -= bias_integrands[:,-1][:,None] # note that expon(q = infinity) = 1
            else:
                bias_integrands = bias_integrands * expon * self.yq**l
                            
            # do FFTLog
            ktemps, bias_ffts = self.sph_kappa.sph(l, bias_integrands)
            ret +=  k**l * interp1d(ktemps, bias_ffts)(k)
                                
        return 4*suppress*np.pi*ret



    def make_table(self, kmin = 1e-3, kmax = 3, nk = 100, func_name = 'power'):
        '''
            Make a table of different terms of P(k), v(k), sigma(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
        '''
        
        if func_name == 'power':
            func = self.p_integrals; iis = np.arange(1+self.num_power_components)
        elif func_name == 'velocity':
            func = self.v_integrals; iis = self.vii
        elif func_name == 'spar':
            func = self.spar_integrals; iis = self.sparii
        elif func_name == 'strace':
            func = self.strace_integrals; iis = self.straceii
        elif func_name == 'gamma1':
            func = self.gamma1_integrals; iis = self.gii
        elif func_name == 'gamma2':
            func = self.gamma2_integrals; iis = self.gii
        elif func_name == 'kappa':
            func = self.kappa_integrals; iis = self.kii

                
        pktable = np.zeros([nk, self.num_power_components+1-1]) # one column for ks, but last column in power now the counterterm
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        pktable[:, 0] = kv[:]
        for foo in range(nk):
            pktable[foo,iis] = func(kv[foo])
        
        return pktable

            
    def make_vtable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.vktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='velocity')

    def make_spartable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.sparktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='spar')

    def make_stracetable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.stracektable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='strace')
        
    def make_gamma1table(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.gamma1ktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='gamma1')
        
    def make_gamma2table(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.gamma2ktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='gamma2')
        
    def make_kappatable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.kappaktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='kappa')
    

    def convert_sigma_bases(self, basis='Legendre'):
        '''
        Function to convert Tr\sigma and \sigma_\par to the desired basis.
        
        These are:
        - Legendre
        
        sigma = sigma_0 delta_ij + sigma_2 (3 k_i k_j - delta_ij)/2
        
        - Polynomial
        
        sigma = sigma_0 delta_ij + sigma_2 k_i k_j
        
        - los (line of sight, note that sigma_0 = kpar and sigma_2 = kperp in this case)
        
        sigma = sigma_0 k_i k_j + sigma_2 (delta_ij - k_i k_j)/2
        
        '''
        if self.sparktable is None or self.stracektable is None:
            print("Error: Need to compute sigma before changing bases!")
            return 0

        kv = self.sparktable[:,0]
        
        if basis == 'Legendre':
            self.s0 = self.stracektable / 3.
            self.s2 = self.sparktable - self.s0
            self.s0[:,0] = kv; self.s2[:,0] = kv

        if basis == 'Polynomial':
            self.s0 = 0.5 * (self.stracektable - self.sparktable)
            self.s2 = 0.5 * (3 * self.sparktable - self.stracektable)
            self.s0[:,0] = kv; self.s2[:,0] = kv

        if basis == 'los':
            self.s0 = self.sparktable
            self.s2 = self.stracektable - self.sparktable
            self.s0[:,0] = kv; self.s2[:,0] = kv


    def convert_gamma_bases(self, basis='Polynomial'):
        '''
        Translates the contraction of gamma into the polynomial/legendre basis
        given by Im[gamma] = g3 \hk_i \hk_j \hk_k + g1 (\hk_i \delta{ij} + et cycl) / 3
        
        '''
        if self.gamma1ktable is None or self.gamma2ktable is None:
            print("Error: Need to compute sigma before changing bases!")
            return 0
                
        kv = self.gamma1ktable[:,0]
        
        # Polynomial basis
        if basis == 'Polynomial':
            self.g1 = 1.5 * self.gamma2ktable - 1.5 * self.gamma1ktable
            self.g3 = 2.5 * self.gamma1ktable - 1.5 * self.gamma2ktable
        
        if basis == 'Legendre':
            self.g1 = 0.6 * self.gamma2ktable
            self.g3 = 2.5 * self.gamma1ktable - 1.5 * self.gamma2ktable
            #self.g1 = self.g1 + 0.6*self.g3
            #self.g3 = 0.4 * self.g3
        
        self.g1[:,0] = kv; self.g3[:,0] = kv
    
    def convert_kappa_bases(self, basis='Polynomial'):
        '''
            Translates the contraction of gamma into the polynomial basis
            given by kappa = kappa0 / 3 * (delta_ij delta_kl + perms) + kappa2 / 6 * (k_i k_j delta_kl + perms)
                               + kappa4 * k_i k_j k_k k_l
        '''
        
        if self.kappaktable is None:
            print("Error: Need to compute kappa before changing bases!")
            return 0

        self.kv = self.kappaktable[:,0]
        
        self.k0 = 3./8 * (self.kappaktable[:,1] - 2*self.kappaktable[:,2] + self.kappaktable[:,3])
        self.k2 = 3./4 * (-self.kappaktable[:,1] + 6*self.kappaktable[:,2] - 5*self.kappaktable[:,3])
        self.k4 = 1./8 * (3*self.kappaktable[:,1] - 30*self.kappaktable[:,2] + 35*self.kappaktable[:,3])


    # the following functions combine all the components into the spectra given some set
    # of bias parameters shared between P(k), v(k), sigma(k)
    # these are, in order, b1, b2, bs, alpha, alpha_v, alpha_s, alpha_s2, sn, sv, s0.

    def combine_bias_terms_vk(self, b1, b2, bs, b3, alpha_v, sv):
        '''
            Combine all the bias terms into one velocity spectrum.
            Assumes the P(k) table has already been computed.
            
            '''
        arr = self.vktable
        
        if self.third_order:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        elif self.shear:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
        else:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2])
        
        try:
            kv = arr[:,0]; za = self.pktable[:,-1]
        except:
            print("Compute the power spectrum table first!")
            
        pktemp = np.copy(arr)[:,1:]
        
        res = np.sum(pktemp * bias_monomials, axis =1) + alpha_v*kv * za + sv*kv
        
        return kv, res

    def combine_bias_terms_sk(self, b1, b2, bs, b3, alpha_s0, alpha_s2, s0_stoch, basis='Polynomial'):
        '''
            Combine all the bias terms into one velocity spectrum.
            Assumes the P(k) table has already been computed.
        '''
        
        self.convert_sigma_bases(basis=basis)
        
        if self.third_order:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        elif self.shear:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
        else:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2])
        
        # Do the monopole
        try:
            arr = self.s0
            kv = arr[:,0]; za = self.pktable[:,-1]
        except:
            print("Compute the power spectrum table first!")
            
        pktemp = np.copy(arr)[:,1:]
        s0 = np.sum(pktemp * bias_monomials, axis =1) + alpha_s0 * za + s0_stoch# here the counterterm is a zero lag and just gives P_Zel
        
        # and the quadratic
        arr = self.s2
        
        kv = arr[:,0]
        pktemp = np.copy(arr)[:,1:]
        
        s2 = np.sum(pktemp * bias_monomials, axis =1) + alpha_s2 * za # there's now a counterterm here too!
        
        return kv, s0 ,s2
        
    def combine_bias_terms_gk(self, b1, b2, bs, b3, alpha_g1, alpha_g3, basis='Polynomial'):
        
        self.convert_gamma_bases(basis=basis)
        
        if self.third_order:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        elif self.shear:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
        else:
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2])
        
        # Do the monopole
        try:
            arr = self.g1
            kv = arr[:,0]; za = self.pktable[:,-1]
        except:
            print("Compute the power spectrum table first!")
            
        pktemp = np.copy(arr)[:,1:]
        g1 = np.sum(pktemp * bias_monomials, axis =1) + alpha_g1 * za / kv# here the counterterm is a zero lag and just gives P_Zel
        
        # and the quadratic
        arr = self.g3
        
        kv = arr[:,0]
        pktemp = np.copy(arr)[:,1:]
        
        g3 = np.sum(pktemp * bias_monomials, axis =1) + alpha_g3 * za / za # there's now a counterterm here too!
        
        return kv, g1, g3
        
    def combine_bias_terms_kk(self, alpha_k2, k0_stoch):
        
        try:
            kv = self.kv
            za = self.pktable[:,-1]
        except:
            print("Compute spectra first!")
        
        return self.kv, self.k0 + k0_stoch, self.k2 + alpha_k2 * za / kv**2, self.k4


