import numpy as np

import time

from scipy.interpolate import interp1d

from Utils.spherical_bessel_transform_fftw import SphericalBesselTransform

from EPT.cleft_kexpanded_fftw import KECLEFT

class KEVelocityMoments(KECLEFT):
    '''
    Class based on cleft_fftw to compute pairwise velocity moments, in expanded LPT.
    
    '''

    def __init__(self, *args, beyond_gauss = True, **kw):
        '''
           Same keywords as the cleft_fftw class. Go look there!
        '''
        
        # Set up the configuration space quantities
        KECLEFT.__init__(self, *args, **kw)
        
        self.setup_onedot()
        self.setup_twodots()
        self.beyond_gauss = beyond_gauss
        
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
            
        self.compute_oneloop_spectra()
    
    def make_tables(self, kmin = 1e-3, kmax = 3, nk = 100, linear_theory=False):
    
        self.kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        self.make_ptable(kmin=kmin, kmax=kmax, nk=nk)
        self.make_vtable(kmin=kmin, kmax=kmax, nk=nk)
        self.make_spartable(kmin=kmin, kmax=kmax, nk=nk)
        self.make_stracetable(kmin=kmin, kmax=kmax, nk=nk)
        self.convert_sigma_bases()
        
        if self.beyond_gauss:
            self.make_gamma1table(kmin=kmin,kmax=kmax,nk=nk)
            self.make_gamma2table(kmin=kmin,kmax=kmax,nk=nk)
            self.convert_gamma_bases()
            
            self.make_kappatable(kmin=kmin,kmax=kmax,nk=nk)
            self.convert_kappa_bases()
        
        
    def compute_oneloop_spectra(self):
        '''
        Compute all velocity spectra nonzero at one loop.
        '''
        self.compute_p_linear()
        self.compute_p_connected()
        self.compute_p_k0()
        self.compute_p_k1()
        self.compute_p_k2()
        self.compute_p_k3()
        self.compute_p_k4()

        self.compute_v_linear()
        self.compute_v_connected()
        self.compute_v_k0()
        self.compute_v_k1()
        self.compute_v_k2()
        self.compute_v_k3()

        self.compute_spar_linear()
        self.compute_spar_connected()
        self.compute_spar_k0()
        self.compute_spar_k1()
        self.compute_spar_k2()

        self.compute_strace_linear()
        self.compute_strace_connected()
        self.compute_strace_k0()
        self.compute_strace_k1()
        self.compute_strace_k2()
        
        if self.beyond_gauss:
        
            self.compute_gamma1_connected()
            self.compute_gamma1_k0()
            self.compute_gamma1_k1()
        
            self.compute_gamma2_connected()
            self.compute_gamma2_k0()
            self.compute_gamma2_k1()
        
            self.compute_kappa_k0()

        
    def update_power_spectrum(self,k,p):
        '''
        Same as the one in cleft_fftw but also do the velocities.
        '''
        super(KEVelocityMoments,self).update_power_spectrum(k,p)
        self.setup_onedot()
        self.setup_twodots()
        self.setup_threedots()

        

    def setup_onedot(self):
        '''
        Create quantities linear in f. All quantities are with f = 1, since converting back is trivial.
        '''
        self.Xdot = self.Xlin; self.sigmadot = self.Xdot[-1]
        self.Ydot = self.Ylin
        
        self.Vdot = 4./3 * self.Vloop #these are only the symmetrized version since all we need...
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
        
        if self.shear or self.third_order:
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
        
        if self.shear or self.third_order:
            self.Xs2ddot = self.Xs2; self.sigmas2ddot = self.Xs2ddot[-1]
            self.Ys2ddot = self.Ys2

    def setup_threedots(self):
        self.Vdddot = 2 * self.Vloop
        self.Tdddot = 2 * self.Tloop

    def compute_v_linear(self):
        
        self.v_linear = np.zeros( (self.num_vel_components, self.N) )
        self.v_linear[0,:] = (- 2 * self.pint )/self.kint
        self.v_linear[1,:] = (- 2 * self.pint )/self.kint

    def compute_v_connected(self):
        
        self.v_connected = np.zeros( (self.num_vel_components, self.N) )
        self.v_connected[0,:] = (- 2 * (2*9./98*self.qf.Q1 + 4*5./21*self.qf.R1) - 12./7*(self.qf.Q2 + 2*self.qf.R2) )/self.kint
        self.v_connected[1,:] = (- 3 * (12*self.qf.R2 + 6*self.qf.Q5 + 6*self.qf.R1)/7 - 3*(10./21*self.qf.R1))/self.kint
        self.v_connected[2,:] = - 12./7 * (self.qf.R1 + self.qf.R2) / self.kint
        self.v_connected[3,:] = - 6./7 * self.qf.Q8 / self.kint
        
        if self.shear or self.third_order:
            self.v_connected[5,:] = - 2 * 2 * 1./7 * self.qf.Qs2 / self.kint
        if self.third_order:
            self.v_connected[7,:] = -2 * self.qf.Rb3 * self.pint / self.kint
    
    def compute_v_k0(self):
        
        self.v_k0 = np.zeros( (self.num_vel_components, self.N) )
        ret = np.zeros(self.num_vel_components)
            
        bias_integrands = np.zeros( (self.num_vel_components,self.N)  )
            
        for l in range(2):
            mu1fac = (l == 1)
                
            bias_integrands[4,:] = mu1fac * (2*self.corlin*self.Udot)
                                       
            if self.shear or self.third_order:
                bias_integrands[6,:] = mu1fac * (2*self.V12dot)
        
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_v.sph(l, bias_integrands)
            self.v_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_v_k1(self):
        
        self.v_k1 = np.zeros( (self.num_vel_components, self.N) )
        ret = np.zeros(self.num_vel_components)
            
        bias_integrands = np.zeros( (self.num_vel_components,self.N)  )
            
        for l in [0,2]:
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
                
            bias_integrands[2,:] = mu0fac * ( self.corlin*self.Xdot ) + \
                                   mu2fac * (2*self.Ulin*self.Udot + self.corlin*self.Ydot)
            bias_integrands[3,:] = mu2fac * (2*self.Ulin*self.Udot)
                                       
            if self.shear or self.third_order:
                bias_integrands[5,:] = mu0fac * (2*self.Xs2dot) + mu2fac * (2*self.Ys2dot)
                #bias_integrands[9,:] = mu0fac * self.zeta
        
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_v.sph(l, bias_integrands)
            self.v_k1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_v_k2(self):
        
        self.v_k2 = np.zeros( (self.num_vel_components, self.N) )
        ret = np.zeros(self.num_vel_components)
            
        bias_integrands = np.zeros( (self.num_vel_components,self.N)  )
            
        for l in range(4):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
                
            bias_integrands[1,:] = mu1fac * ( -2*self.Ulin*self.Xdot - self.Udot*self.Xlin ) + \
                                   mu3fac * ( -2*self.Ulin*self.Ydot - self.Udot*self.Ylin )
                                       
            #if self.shear or self.third_order:
                #bias_integrands[8,:] = mu0fac * self.chi
                #bias_integrands[9,:] = mu0fac * self.zeta
        
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_v.sph(l, bias_integrands)
            self.v_k2 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_v_k3(self):
        
        self.v_k3 = np.zeros( (self.num_vel_components, self.N) )
        ret = np.zeros(self.num_vel_components)
            
        bias_integrands = np.zeros( (self.num_vel_components,self.N)  )
            
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
                
            bias_integrands[0,:] = mu0fac * ( - 0.5*self.Xdot*self.Xlin ) + \
                                   mu2fac * ( - 0.5*(self.Xdot*self.Ylin+self.Ydot*self.Xlin) ) + \
                                   mu4fac * ( - 0.5*self.Ylin*self.Ydot )
                                       
            #if self.shear or self.third_order:
                #bias_integrands[8,:] = mu0fac * self.chi
                #bias_integrands[9,:] = mu0fac * self.zeta
        
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_v.sph(l, bias_integrands)
            self.v_k3 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_spar_linear(self):
        self.spar_linear= np.zeros( (self.num_spar_components, self.N) )
        self.spar_linear[0,:] = (-2 * self.pint)/self.kint**2
    
    def compute_spar_connected(self):
        self.spar_connected = np.zeros( (self.num_spar_components, self.N) )
        self.spar_connected[0,:] = (-2*(4*9./98*self.qf.Q1 + 6*5./21*self.qf.R1) - 6*(5./3*3./7*(self.qf.Q2+2*self.qf.R2)))/self.kint**2
        self.spar_connected[1,:] = (-2 * 2 * (12./7*self.qf.R2 + 6./7*self.qf.Q5 + 6./7*self.qf.R1))/self.kint**2

    def compute_spar_k0(self):
        self.spar_k0 = np.zeros( (self.num_spar_components, self.N) )
        bias_integrands = np.zeros( (self.num_spar_components,self.N)  )
        
        for l in range(3):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            
            bias_integrands[2,:] = mu0fac * (self.corlin*self.Xddot) + mu2fac * (self.corlin*self.Yddot + 2*self.Udot**2)
            bias_integrands[3,:] = mu2fac * (2*self.Udot**2)

            if self.shear or self.third_order:
                bias_integrands[4,:] = mu0fac * (2*self.Xs2ddot) + mu2fac * (2*self.Ys2ddot)
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_spar.sph(l, bias_integrands)
            self.spar_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def compute_spar_k1(self):
        self.spar_k1 = np.zeros( (self.num_spar_components, self.N) )
        bias_integrands = np.zeros( (self.num_spar_components,self.N)  )
        
        for l in range(4):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            
            bias_integrands[1,:] = mu1fac * (-2*(self.Ulin*self.Xddot + 2*self.Udot*self.Xdot) ) + \
                                   mu3fac * (-2*(self.Ulin*self.Yddot + 2*self.Udot*self.Ydot) )
                                   
            #if self.shear or self.third_order:
                #bias_integrands[8,:] = mu0fac * self.chi
                #bias_integrands[9,:] = mu0fac * self.zeta
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_spar.sph(l, bias_integrands)
            self.spar_k1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
        
    def compute_spar_k2(self):
        self.spar_k2 = np.zeros( (self.num_spar_components, self.N) )
        bias_integrands = np.zeros( (self.num_spar_components,self.N)  )
        
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            
            bias_integrands[0,:] = mu0fac * (-self.Xdot**2 - 0.5*self.Xddot*self.Xlin) + \
                                   mu2fac * (- 2*self.Xdot*self.Ydot - 0.5*(self.Xddot*self.Ylin + self.Yddot*self.Xlin)) + \
                                   mu4fac * (-self.Ydot**2 - 0.5*self.Yddot*self.Ylin)
                                   
            #if self.shear or self.third_order:
                #bias_integrands[8,:] = mu0fac * self.chi
                #bias_integrands[9,:] = mu0fac * self.zeta
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_spar.sph(l, bias_integrands)
            self.spar_k2 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_strace_linear(self):
        self.strace_linear = np.zeros( (self.num_spar_components, self.N) )
        self.strace_linear[0,:] = (-2 * self.pint )/self.kint**2
    
    def compute_strace_connected(self):
        self.strace_connected = np.zeros( (self.num_spar_components, self.N) )
        self.strace_connected[0,:] = (- 2*(4*9./98*self.qf.Q1 + 6*5./21*self.qf.R1) \
                                        + 6./7*(self.qf.Q1-5*self.qf.Q2+4*self.qf.R1-10*self.qf.R2))/self.kint**2
        self.strace_connected[1,:] = (-4/self.kint**2* 3./7* (4*self.qf.R2+2*self.qf.Q5) )
    
    def compute_strace_k0(self):
        self.strace_k0 = np.zeros( (self.num_strace_components, self.N) )
        bias_integrands = np.zeros( (self.num_strace_components,self.N)  )
        
        for l in range(self.jn):
            mu0fac = (l == 0)
            
            bias_integrands[2,:] = mu0fac * (self.corlin*(3*self.Xddot + self.Yddot) + 2*self.Udot**2)
            bias_integrands[3,:] = mu0fac * (2 * self.Udot**2)
            
            if self.shear or self.third_order:
                bias_integrands[4,:] = mu0fac * ( 2*(3*self.Xs2ddot + self.Ys2ddot) )

    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_strace.sph(l, bias_integrands)
            self.strace_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_strace_k1(self):
        self.strace_k1 = np.zeros( (self.num_strace_components, self.N) )
        bias_integrands = np.zeros( (self.num_strace_components,self.N)  )
        
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            
            bias_integrands[1,:] = mu1fac * (-2*self.Ulin*(3*self.Xddot+self.Yddot) - 4*self.Udot*(self.Xdot+self.Ydot) )
                                       
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_strace.sph(l, bias_integrands)
            self.strace_k1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_strace_k2(self):
        self.strace_k2 = np.zeros( (self.num_strace_components, self.N) )
        bias_integrands = np.zeros( (self.num_strace_components,self.N)  )
        
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu1fac = (l == 1)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)

            bias_integrands[0,:] = mu0fac * ( (3*self.Xddot + self.Yddot)*(- 0.5*self.Xlin) - self.Xdot**2) + \
                                   mu2fac * ((3*self.Xddot + self.Yddot)*(- 0.5*self.Ylin) - (self.Ydot**2+2*self.Xdot*self.Ydot))
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_strace.sph(l, bias_integrands)
            self.strace_k2 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_gamma1_connected(self):
        self.gamma1_connected = np.zeros( (self.num_gamma_components, self.N) )
        self.gamma1_connected[0,:] = 36./7 * (self.qf.Q2 + 2*self.qf.R2) / self.kint**3
        
    def compute_gamma1_k0(self):
        self.gamma1_k0 = np.zeros( (self.num_gamma_components, self.N) )
        bias_integrands = np.zeros( (self.num_gamma_components,self.N)  )
        
        for l in range(self.jn):
            mu1fac = (l == 1)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            
            bias_integrands[1,:] = mu1fac * (6*self.Udot*self.Xddot) + mu3fac * (6*self.Udot*self.Yddot)
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_gamma1.sph(l, bias_integrands)
            self.gamma1_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
        
    def compute_gamma1_k1(self):
        self.gamma1_k1 = np.zeros( (self.num_gamma_components, self.N) )
        bias_integrands = np.zeros( (self.num_gamma_components,self.N)  )
        
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
            
            bias_integrands[0,:] = mu0fac * (3*self.Xdot*self.Xddot) + \
                                   mu2fac * (3*(self.Xdot*self.Yddot+self.Ydot*self.Xddot)) + \
                                   mu4fac * (3*self.Ydot*self.Yddot)
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_gamma1.sph(l, bias_integrands)
            self.gamma1_k1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_gamma2_connected(self):
        self.gamma2_connected = np.zeros( (self.num_gamma_components, self.N) )
        self.gamma2_connected[0,:] = - 12./7 * (2*self.qf.R1 - 6*self.qf.R2 + self.qf.Q1 - 3*self.qf.Q2)/self.kint**3
     
    def compute_gamma2_k0(self):
        self.gamma2_k0 = np.zeros( (self.num_gamma_components, self.N) )
        bias_integrands = np.zeros( (self.num_gamma_components,self.N)  )
        
        for l in range(self.jn):
            mu1fac = (l == 1)
            mu3fac = 0.6 * (l==1) - 0.4 * (l==3)
            
            bias_integrands[1,:] = mu1fac * ( 2*self.Udot*(5*self.Xddot + 3*self.Yddot) )
    
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
        
            # do FFTLog
            ktemps, bias_ffts = self.sph_gamma2.sph(l, bias_integrands)
            self.gamma2_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    def compute_gamma2_k1(self):
        self.gamma2_k1 = np.zeros( (self.num_gamma_components, self.N) )
        bias_integrands = np.zeros( (self.num_gamma_components,self.N)  )
            
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
                
            bias_integrands[0,:] = mu0fac * ( (5*self.Xdot*self.Xddot+self.Xdot*self.Yddot) ) + \
                                       mu2fac * ( (2*self.Xdot*self.Yddot+self.Ydot*(5*self.Xddot+3*self.Yddot)) )
        
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_gamma2.sph(l, bias_integrands)
            self.gamma2_k1 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)
    
    
    def compute_kappa_k0(self):
        self.kappa_k0 = np.zeros( (self.num_kappa_components, self.N) )
        bias_integrands = np.zeros( (self.num_kappa_components,self.N)  )
            
        for l in range(self.jn):
            mu0fac = (l == 0)
            mu2fac = 1./3 * (l==0) - 2./3*(l==2)
            mu4fac = 0.2 * (l==0) - 4./7*(l==2) + 8./35*(l==4)
                
            bias_integrands[0,:] = mu0fac * (15 * self.Xddot**2 + 10 * self.Xddot*self.Yddot + 3 * self.Yddot**2)
            bias_integrands[1,:] = mu0fac * (5 * self.Xddot**2 + self.Xddot*self.Yddot) + \
                                   mu2fac * (7*self.Xddot*self.Yddot + 3*self.Yddot**2)
            bias_integrands[2,:] = mu0fac * (3 * self.Xddot**2) + mu2fac * (6*self.Xddot*self.Yddot) + mu4fac * (3*self.Yddot**2)
        
            if l >= 0:
                bias_integrands -= bias_integrands[:,-1][:,None]
            
            # do FFTLog
            ktemps, bias_ffts = self.sph_kappa.sph(l, bias_integrands)
            self.kappa_k0 += 4 * np.pi * interp1d(ktemps, bias_ffts, bounds_error=False)(self.kint)

    def make_table(self, kmin = 1e-3, kmax = 3, nk = 100, func_name = 'power', linear_theory=False):
        '''
            Make a table of different terms of P(k), v(k), sigma(k) between a given
            'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
            This is the most time consuming part of the code.
        '''
        
        pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks, but last column in power now the counterterm
        kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        pktable[:, 0] = kv[:]
        
        if not linear_theory:
            if func_name == 'power':
                components = [ (1, self.p_linear+self.p_connected + self.p_k0), (self.kint, self.p_k1),\
                           (self.kint**2, self.p_k2), (self.kint**3, self.p_k3), (self.kint**4, self.p_k4)  ]
                iis = np.arange(1,1+self.num_power_components)
            elif func_name == 'velocity':
                components = [ (1, self.v_linear+self.v_connected+self.v_k0), (self.kint, self.v_k1), (self.kint**2, self.v_k2), (self.kint**3, self.v_k3)  ]
                iis = self.vii
            elif func_name == 'spar':
                components = [ (1, self.spar_linear+self.spar_connected+self.spar_k0),(self.kint, self.spar_k1),(self.kint**2, self.spar_k2)  ]
                iis = self.sparii
            elif func_name == 'strace':
                components = [ (1, self.strace_linear+self.strace_connected+self.strace_k0),(self.kint, self.strace_k1),(self.kint**2, self.strace_k2)  ]
                iis = self.straceii
            elif func_name == 'gamma1':
                components = [ (1, self.gamma1_connected+self.gamma1_k0),(self.kint, self.gamma1_k1)  ]
                iis = self.gii
            elif func_name == 'gamma2':
                components = [ (1, self.gamma2_connected+self.gamma2_k0),(self.kint, self.gamma2_k1)  ]
                iis = self.gii
            elif func_name == 'kappa':
                components = [ (1, self.kappa_k0)  ]
                iis = self.kii
        else:
            if func_name == 'power':
                components = [ (1, self.p_linear) ]
                iis = np.arange(1,1+self.num_power_components)
            elif func_name == 'velocity':
                components = [ (1, self.v_linear)  ]
                iis = self.vii
            elif func_name == 'spar':
                components = [ (1, self.spar_linear)  ]
                iis = self.sparii
            elif func_name == 'strace':
                components = [ (1, self.strace_linear)  ]
                iis = self.straceii
            elif func_name == 'gamma1':
                return pktable
            elif func_name == 'gamma2':
                return pktable
            elif func_name == 'kappa':
                return pktable
                
        # sum the components:
        ptable = 0
        for (kpow, comp) in components:
            ptable += kpow * comp
        
        # interpolate onto range of interest
        for jj in range(len(iis)):
            pktable[:,iis[jj]] = interp1d(self.kint, ptable[jj,:])(kv)
        
        return pktable
    
    def make_ptable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.pktable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='power',linear_theory=True)
        self.pktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='power')
    
    def make_vtable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.vktable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='velocity',linear_theory=True)
        self.vktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='velocity')

    def make_spartable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.sparktable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='spar',linear_theory=True)
        self.sparktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='spar')

    def make_stracetable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.stracektable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='strace',linear_theory=True)
        self.stracektable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='strace')
        
    def make_gamma1table(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.gamma1ktable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='gamma1',linear_theory=True)
        self.gamma1ktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='gamma1')
        
    def make_gamma2table(self, kmin = 1e-3, kmax = 3, nk = 100, linear_theory=True):
        self.gamma2ktable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='gamma2', linear_theory=True)
        self.gamma2ktable = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='gamma2')
        
    def make_kappatable(self, kmin = 1e-3, kmax = 3, nk = 100):
        self.kappaktable_linear = self.make_table(kmin=kmin,kmax=kmax,nk=nk,func_name='kappa',linear_theory=True)
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
            self.s0_linear = self.stracektable_linear / 3.
            self.s2_linear = self.sparktable_linear - self.s0_linear
            self.s0_linear[:,0] = kv; self.s2_linear[:,0] = kv
        
            self.s0 = self.stracektable / 3.
            self.s2 = self.sparktable - self.s0
            self.s0[:,0] = kv; self.s2[:,0] = kv

        if basis == 'Polynomial':
            self.s0_linear = 0.5 * (self.stracektable_linear - self.sparktable_linear)
            self.s2_linear = 0.5 * (3 * self.sparktable_linear - self.stracektable_linear)
            self.s0_linear[:,0] = kv; self.s2_linear[:,0] = kv
        
            self.s0 = 0.5 * (self.stracektable - self.sparktable)
            self.s2 = 0.5 * (3 * self.sparktable - self.stracektable)
            self.s0[:,0] = kv; self.s2[:,0] = kv

        if basis == 'los':
            self.s0_linear = self.sparktable_linear
            self.s2_linear = self.stracektable_linear - self.sparktable_linear
            self.s0_linear[:,0] = kv; self.s2_linear[:,0] = kv
        
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

        
        self.k0 = 3./8 * (self.kappaktable[:,1] - 2*self.kappaktable[:,2] + self.kappaktable[:,3])
        self.k2 = 3./4 * (-self.kappaktable[:,1] + 6*self.kappaktable[:,2] - 5*self.kappaktable[:,3])
        self.k4 = 1./8 * (3*self.kappaktable[:,1] - 30*self.kappaktable[:,2] + 35*self.kappaktable[:,3])


    # the following functions combine all the components into the spectra given some set
    # of bias parameters shared between P(k), v(k), sigma(k)
    # these are, in order, b1, b2, bs, alpha, alpha_v, alpha_s, alpha_s2, sn, sv, s0.

    def combine_bias_terms_vk(self, bvec):
        '''
            Combine all the bias terms into one velocity spectrum.
            Assumes the P(k) table has already been computed.
            
            '''
        arr = self.vktable
        
        if self.third_order:
            b1, b2, bs, b3, alpha, alpha_v, alpha_s, alpha_s2, sn, sv, s0 = bvec # only alpha and sn are relevant here
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        elif self.shear:
            b1, b2, bs, alpha, alpha_v, alpha_s, alpha_s2, sn, sv, s0 = bvec # only alpha and sn are relevant here
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
        else:
            b1, b2, alpha, alpha_v, alpha_s, alpha_s2, sn, sv, s0 = bvec # only alpha and sn are relevant here
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2])
        
        try:
            kv = arr[:,0]; za = self.pktable[:,-1]
        except:
            print("Compute the power spectrum table first!")
            
        pktemp = np.copy(arr)[:,1:]
        
        res = np.sum(pktemp * bias_monomials, axis =1) + alpha_v*kv * za + sv*kv
        
        return kv, res

    def combine_bias_terms_sk(self, bvec, basis='Legendre'):
        '''
            Combine all the bias terms into one velocity spectrum.
            Assumes the P(k) table has already been computed.
            '''
        
        self.convert_sigma_bases(basis=basis)
        
        if self.third_order:
            b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, s0 = bvec # only alpha and sn are relevant here
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        elif self.shear:
            b1, b2, bs, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, s0 = bvec # only alpha and sn are relevant here
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2])
        else:
            b1, b2, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, s0 = bvec # only alpha and sn are relevant here
            bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2])
        
        # Do the constant coefficient
        arr = self.s0
        
        try:
            kv = arr[:,0]; za = self.pktable[:,-1]
        except:
            print("Compute the power spectrum table first!")
            
        pktemp = np.copy(arr)[:,1:]
        
        s0 = np.sum(pktemp * bias_monomials, axis =1) + alpha_s0 * za + s0# here the counterterm is a zero lag and just gives P_Zel
        
        # and the quadratic
        arr = self.s2
        
        kv = arr[:,0]
        pktemp = np.copy(arr)[:,1:]
        
        s2 = np.sum(pktemp * bias_monomials, axis =1) + alpha_s2 * za # there's now a counterterm here too!
        
        return kv, s0 ,s2


