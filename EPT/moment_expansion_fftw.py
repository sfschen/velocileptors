import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


from Utils.loginterp import loginterp
from EPT.ept_fftw import EPT

class MomentExpansion:

    '''
    Class to compute IR-resummed velocity moments and RSD using the moment expansion appraoch in EPT.
    
    Based on the "kexpanded" classes, with the main new feature being that everything is calculated twice,
    once for the usual linear power spectrum and once for a no-wiggle version that can be input.
    
    Contains functions to assemble spectra given bias parameters, counterterms etc.
    
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
        self.plin_ir = self.plin_nw + self.plin_w * self.damp_exp
        
        
        self.pktable_nw = self.ept_nw.pktable_ept
        self.pktable_w  = self.damp_fac[:,None] *( (self.ept.pktable_ept - self.pktable_nw) - self.damp_exp[:,None] * (self.ept.pktable_ept_linear - self.ept_nw.pktable_ept_linear) )
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
        
    
    # The following classes combine the bias terms into velocity moments given some bias parameters
    # See the expressions in Chen, Vlah and White (2020) for further details.
        
    def combine_bias_terms_pk(self,b1,b2,bs,b3,alpha,sn):
        
        return b1**2 * self.pktable[:,1] + b1*b2 * self.pktable[:,2] + b1*bs * self.pktable[:,3] \
                + b2**2 * self.pktable[:,4] + b2*bs * self.pktable[:,5] + bs**2 * self.pktable[:,6] \
                + b1*b3 * self.pktable[:,7] + alpha*self.pktable[:,8] + sn
                
    def combine_bias_terms_vk(self,b1,b2,bs,b3,alphav, sv):
        
        return b1 * self.vktable[:,1] + b1**2 * self.vktable[:,2] + b2 * self.vktable[:,3] \
              +b1*b2*self.vktable[:,4] + bs*self.vktable[:,5] + b1*bs * self.vktable[:,6] \
              +b3*self.vktable[:,7] + alphav * self.vktable[:,8] + sv * self.vktable[:,0]
              
    def combine_bias_terms_sk(self,b1,b2,bs,b3,alpha0,alpha2,sigma0):
        
        s0 = self.s0ktable[:,1] + b1 * self.s0ktable[:,2] + b1**2 * self.s0ktable[:,3] \
            + b2 * self.s0ktable[:,4] + bs * self.s0ktable[:,5] + alpha0*self.s0ktable[:,6] + sigma0
            
        s2 = self.s2ktable[:,1] + b1 * self.s2ktable[:,2] + b1**2 * self.s2ktable[:,3] \
        + b2 * self.s2ktable[:,4] + bs * self.s2ktable[:,5] + alpha2*self.s2ktable[:,6]
        
        return s0, s2
        
        
    def combine_bias_terms_gk(self,b1,b2,bs,b3,alpha1, alpha3):
    
        g1 = self.g1ktable[:,1] + b1*self.g1ktable[:,2] + alpha1*self.g1ktable[:,-1]
        g3 = self.g3ktable[:,1] + b1*self.g3ktable[:,2] + alpha3*self.g1ktable[:,-1]
        
        return g1, g3
        
    def combine_bias_terms_kk(self,b1,b2,bs,b3,alpha2,stoch_k0):
    
        k0 = self.k0 + stoch_k0
        k2 = self.k2 + alpha2*self.pktable[:,-1] / self.pktable[:,0]**2
        k4 = self.k4
        
        return k0, k2, k4
    
    # The following functions combine the bias parameters and velocity moments into redshift-space power spectra
    # If beyond_gauss is false use the first two velocity moments (v(k) and sigma(k)) plus the counterterm ansatz for the third moment.
    # In this case the parameters alpha_g1, alpha_g3, alpha_k2, stoch_k0 are not used--set to zero if desired.
    # Otherwise gives the full moment expansion expression up to one-loop order.
    
    def compute_redshift_space_power_at_mu(self,pars,f,mu_obs,counterterm_c3=0,beyond_gauss=False,reduced=True,apar=1.0,aperp=1.0):
        '''
        If AP parameters apar and aperp are nonzero then the input/output
        k, mu refer to the observed.
        We use "physical" AP parameters, defined as the scaling of distances
        parallel and perpendicular to the line of sight.
        '''
        # Change mu to the "true" from the input observed
        # Note that kv below refers to "true" k
        # We follow the notation/conventions in
        # https://arxiv.org/abs/1312.4611  Eqs. (58-60).
        F = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F**2 - 1) )
        mu = mu_obs / F / AP_fac

        
        if beyond_gauss:
            if reduced:
                b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
                alpha, alphav, alpha_s0, alpha_s2, alpha_g1, alpha_g3, alpha_k2 = alpha0, alpha2, 0, alpha4, 0, 0, alpha6
                sn, sv, sigma0, stoch_k0 = sn, sn2, 0, sn4
            else:
                b1,b2,bs,b3,alpha,alphav,alpha_s0,alpha_s2,alpha_g1,alpha_g3,alpha_k2,sn,sv,sigma0,stoch_k0 = pars
        else:
            if reduced:
                b1, b2, bs, b3, alpha0, alpha2, alpha4, sn, sn2 = pars
                alpha, alphav, alpha_s0, alpha_s2 = alpha0, alpha2, 0, alpha4
                sn, sv, sigma0 = sn, sn2, 0
            else:
                b1,b2,bs,b3,alpha,alphav,alpha_s0,alpha_s2,sn,sv,sigma0 = pars
        
        kv = self.pktable[:,0]
        mu2 = mu**2
        
        pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha,sn)
        vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alphav,sv)
        s0k, s2k = self.combine_bias_terms_sk(b1,b2,bs,b3,alpha_s0,alpha_s2,sigma0)
        
        ret = pk - f*kv*mu2*vk - 0.5*f**2*kv**2*mu2 * (s0k + 0.5*s2k*(3*mu2-1) )
        
        if beyond_gauss:
            g1k, g3k = self.combine_bias_terms_gk(b1,b2,bs,b3,alpha_g1,alpha_g3)
            k0k, k2k, k4k = self.combine_bias_terms_kk(b1,b2,bs,b3,alpha_k2,stoch_k0)
            ret += 1./6 * f**3 * (kv*mu)**3 * (g1k * mu + g3k * mu**3)\
                   + 1./24 * f**4 * (kv*mu)**4 * (k0k + k2k * mu2 + k4k * mu2**2)
        else:
            ret += 1./6 * counterterm_c3 * kv**2 * mu**4 * self.plin_ir

        # Interpolate onto true wavenumbers
        kobs = self.kv * aperp / AP_fac
        pobs = interp1d(kobs,ret,kind='cubic',fill_value='extrapolate')(self.kv)
        pobs = pobs / aperp**2 / apar
        
        return kv, pobs
        
    def compute_redshift_space_power_multipoles(self, pars, f, counterterm_c3=0, ngauss=4, reduced=False,beyond_gauss=False):

        # Generate the sampling
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),self.nk))
        
        for ii, nu in enumerate(nus_calc):
            self.pknutable[ii,:] = self.compute_redshift_space_power_at_mu(pars,f,nu,beyond_gauss=beyond_gauss,reduced=reduced,counterterm_c3=counterterm_c3)[1]
                
        self.pknutable[ngauss:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None]*self.pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None]*self.pknutable,axis=0)
        
        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable
                
                
