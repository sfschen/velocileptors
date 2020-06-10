import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from Utils.loginterp import loginterp

from EPT.ept_fftw import EPT

class REPT:

    '''
    Class to compute IR-resummed RSD power spectrumusing the moment expansion appraoch in EPT.
    Instead of summing the velocity moments separately, the full 1-loop power spectrum with
    linear velocities resummed is computed.
    
    Based on the EPT class.
    
    '''
    
    def __init__(self, k, p, pnw=None, *args, rbao = 110, kmin = 1e-2, kmax = 0.5, nk = 100, **kw):
        
        self.nk, self.kmin, self.kmax = nk, kmin, kmax
        self.rbao = rbao
        
        self.ept = EPT( k, p, kmin=kmin, kmax=kmax, **kw)
        
        if pnw is None:
            knw = self.ept.kint
            Nfilter =  np.ceil(np.log(7) /  np.log(knw[-1]/knw[-2])) // 2 * 2 + 1 # filter length ~ log span of one oscillation from k = 0.01
            print(Nfilter)
            pnw = savgol_filter(self.ept.pint, int(Nfilter), 4)
        else:
            knw, pnw = k, pnw
        self.ept_nw = EPT( knw, pnw, kmin=kmin, kmax=kmax, **kw)
        
        self.beyond_gauss = self.ept.beyond_gauss
        
        self.kv = self.ept.kv
        self.plin  = loginterp(k, p)(self.kv)
        self.plin_nw = loginterp(knw, pnw)(self.kv)
        self.plin_w = self.plin - self.plin_nw
        self.sigma_squared_bao = np.interp(self.rbao, self.ept_nw.qint, self.ept_nw.Xlin + self.ept_nw.Ylin/3.)
        self.damp_exp = - 0.5 * self.kv**2 * self.sigma_squared_bao
        self.damp_fac = np.exp(self.damp_exp)
        
        
        self.pktable_nw = self.ept_nw.pktable_ept
        self.pktable_w  =  self.ept.pktable_ept - self.pktable_nw
        self.pktable_w[:,0] = self.kv
        self.pktable = self.pktable_nw + self.pktable_w; self.pktable[:,0] = self.kv
        
        self.vktable_nw = self.ept_nw.vktable_ept
        self.vktable_w  = self.damp_fac[:,None] *( (self.ept.vktable_ept - self.vktable_nw) - self.damp_exp[:,None] * (self.ept.vktable_ept_linear - self.ept_nw.vktable_ept_linear) )
        self.vktable_w[:,0] = self.kv
        self.vktable = self.vktable_nw + self.vktable_w; self.vktable[:,0] = self.kv
        
        self.s0ktable_nw = self.ept_nw.s0ktable_ept
        self.s0ktable_w  = self.damp_fac[:,None] *( (self.ept.s0ktable_ept - self.s0ktable_nw) - self.damp_exp[:,None] * (self.ept.s0ktable_ept_linear - self.ept_nw.s0ktable_ept_linear) )
        self.s0ktable_w[:,0] = self.kv
        self.s0ktable = self.s0ktable_nw + self.s0ktable_w; self.s0ktable[:,0] = self.kv
        
        self.s2ktable_nw = self.ept_nw.s2ktable_ept
        self.s2ktable_w  = self.damp_fac[:,None] *( (self.ept.s2ktable_ept - self.s2ktable_nw) - self.damp_exp[:,None] * (self.ept.s2ktable_ept_linear - self.ept_nw.s2ktable_ept_linear) )
        self.s2ktable_w[:,0] = self.kv
        self.s2ktable = self.s2ktable_nw + self.s2ktable_w; self.s2ktable[:,0] = self.kv
        
        if self.beyond_gauss:
            self.g1ktable_nw = self.ept_nw.g1ktable_ept
            self.g1ktable_w = self.damp_fac[:,None] * (self.ept.g1ktable_ept - self.ept_nw.g1ktable_ept)
            self.g1ktable_w[:,0] = self.kv
            self.g1ktable = self.g1ktable_nw + self.g1ktable_w; self.g1ktable[:,0] = self.kv
        
            self.g3ktable_nw = self.ept_nw.g3ktable_ept
            self.g3ktable_w = self.damp_fac[:,None] * (self.ept.g3ktable_ept - self.ept_nw.g3ktable_ept)
            self.g3ktable_w[:,0] = self.kv
            self.g3ktable = self.g3ktable_nw + self.g3ktable_w; self.g3ktable[:,0] = self.kv
        
            self.k0_nw, self.k2_nw, self.k4_nw = self.ept_nw.k0, self.ept_nw.k2, self.ept_nw.k4
            self.k0_w = self.damp_fac * (self.ept.k0 - self.ept_nw.k0)
            self.k2_w = self.damp_fac * (self.ept.k2 - self.ept_nw.k2)
            self.k4_w = self.damp_fac * (self.ept.k4 - self.ept_nw.k4)
            self.k0 = self.k0_nw + self.k0_w; self.k2 = self.k2_nw + self.k2_w; self.k4 = self.k4_nw + self.k4_w
        
    
    # Combine everything into redshift space power spectrum, all at once!
    
    def compute_redshift_space_power_at_mu(self, pars, f, mu_obs, apar=1., aperp=1.):
        
        # Change mu to the "true" from the input observed
        # Note that kv below refers to "true" k
        # We follow the notation/conventions in
        # https://arxiv.org/abs/1312.4611  Eqs. (58-60).
        F = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F**2 - 1) )
        mu = mu_obs / F / AP_fac
        
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        
        kv = self.kv
        ret = 0
        damp_exp = self.damp_exp * (1 + f*(2+f)*mu**2)
        damp_fac = np.exp(damp_exp)
        
        # Assemble in terms of powers of f (unresummed)
        
        pk_nw = b1**2 * self.pktable_nw[:,1] + b1*b2 * self.pktable_nw[:,2] + b1*bs * self.pktable_nw[:,3] \
        + b2**2 * self.pktable_nw[:,4] + b2*bs * self.pktable_nw[:,5] + bs**2 * self.pktable_nw[:,6] \
        + b1*b3 * self.pktable_nw[:,7]
        pk_w  = b1**2 * self.pktable_w[:,1] + b1*b2 * self.pktable_w[:,2] + b1*bs * self.pktable_w[:,3] \
        + b2**2 * self.pktable_w[:,4] + b2*bs * self.pktable_w[:,5] + bs**2 * self.pktable_w[:,6] \
        + b1*b3 * self.pktable_w[:,7]
        
        vk_nw = b1 * self.vktable_nw[:,1] + b1**2 * self.vktable_nw[:,2] + b2 * self.vktable_nw[:,3] \
        +b1*b2*self.vktable_nw[:,4] + bs*self.vktable_nw[:,5] + b1*bs * self.vktable_nw[:,6] \
        +b3*self.vktable_nw[:,7]
        vk_w = b1 * self.vktable_w[:,1] + b1**2 * self.vktable_w[:,2] + b2 * self.vktable_w[:,3] \
        +b1*b2*self.vktable_w[:,4] + bs*self.vktable_w[:,5] + b1*bs * self.vktable_w[:,6] \
        +b3*self.vktable_w[:,7]
        
        s0_nw = self.s0ktable_nw[:,1] + b1 * self.s0ktable_nw[:,2] + b1**2 * self.s0ktable_nw[:,3] \
            + b2 * self.s0ktable_nw[:,4] + bs * self.s0ktable_nw[:,5]
        s0_w = self.s0ktable_w[:,1] + b1 * self.s0ktable_w[:,2] + b1**2 * self.s0ktable_w[:,3] \
        + b2 * self.s0ktable_w[:,4] + bs * self.s0ktable_w[:,5]
            
        s2_nw = self.s2ktable_nw[:,1] + b1 * self.s2ktable_nw[:,2] + b1**2 * self.s2ktable_nw[:,3] \
        + b2 * self.s2ktable_nw[:,4] + bs * self.s2ktable_nw[:,5] + alpha2*self.s2ktable_nw[:,6]
        s2_w = self.s2ktable_w[:,1] + b1 * self.s2ktable_w[:,2] + b1**2 * self.s2ktable_w[:,3] \
        + b2 * self.s2ktable_w[:,4] + bs * self.s2ktable_w[:,5] + alpha2*self.s2ktable_w[:,6]
        
        g1_nw = self.g1ktable_nw[:,1] + b1*self.g1ktable_nw[:,2]
        g1_w  = self.g1ktable_w[:,1] + b1*self.g1ktable_w[:,2]
        
        g3_nw = self.g3ktable_nw[:,1] + b1*self.g3ktable_nw[:,2]
        g3_w  = self.g3ktable_w[:,1] + b1*self.g3ktable_w[:,2]
        
        k0_nw, k2_nw, k4_nw = self.k0_nw, self.k2_nw, self.k4_nw
        k0_w, k2_w, k4_w = self.k0_w, self.k2_w, self.k4_w
        
        # Now add them all together!
        # no wiggle
        ret += pk_nw - f*kv*mu**2*vk_nw - 0.5*f**2*kv**2*mu**2 * (s0_nw + 0.5*s2_nw*(3*mu**2-1) ) + \
              + 1./6 * f**3 * (kv*mu)**3 * (g1_nw * mu + g3_nw * mu**3)\
              + 1./24 * f**4 * (kv*mu)**4 * (k0_nw + k2_nw * mu**2 + k4_nw * mu**4)
              
        # wiggle
        ret += damp_fac * (pk_w - f*kv*mu**2*vk_w - 0.5*f**2*kv**2*mu**2 * (s0_w + 0.5*s2_w*(3*mu**2-1) ) + \
        + 1./6 * f**3 * (kv*mu)**3 * (g1_w * mu + g3_w * mu**3)\
        + 1./24 * f**4 * (kv*mu)**4 * (k0_w + k2_w * mu**2 + k4_w * mu**4) )
        
        # linear theory compensation
        ret -= damp_exp * damp_fac * (b1 + f*mu**2)**2 * self.plin_w
        
        # counterterms and stochastic terms
        ret += (alpha0 + alpha2 * mu**2 + alpha4 * mu**4 + alpha6 * mu**6) * kv**2 * (self.plin_nw + damp_fac*self.plin_w)
        ret += (sn + kv**2 * mu**2 * sn2 + kv**4 * mu**4 * sn4)
        
        # Interpolate onto true wavenumbers
        kobs = kv * aperp / AP_fac
        pobs = interp1d(kobs,ret,kind='cubic',fill_value='extrapolate')(self.kv)
        pobs = pobs / aperp**2 / apar
        
        return kv, pobs
        


    def compute_redshift_space_power_multipoles(self, pars, f, ngauss=4, apar=1., aperp=1.):

        # Generate the sampling
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),self.nk))
        
        for ii, nu in enumerate(nus_calc):
            self.pknutable[ii,:] = self.compute_redshift_space_power_at_mu(pars,f,nu,apar=apar,aperp=aperp)[1]
                
        self.pknutable[ngauss:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None]*self.pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None]*self.pknutable,axis=0)
        
        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable
