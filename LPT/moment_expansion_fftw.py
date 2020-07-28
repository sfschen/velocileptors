import numpy as np
from scipy.interpolate import interp1d

from Utils.loginterp import loginterp
from LPT.velocity_moments_fftw import VelocityMoments



class MomentExpansion(VelocityMoments):
    '''
    Class to calculate the redshift space power spectrum in the
    moment expansion approach.
    
    Inherits the VelocityMoments class which itself inherits the CLEFT class.
    
    Can Operate in two main modes:
    (1) If beyond_gauss is set to true, this computes the full 1-loop
        redshift space power spectrum in LPT
    (2) Otherwise, it uses the counterterm ansatz for the third moment,
        shown to be an excellent approximaiton compared to data.
    
    Also, for both (1) and (2) one can choose the full basis of
    bias/counterterm/stochastic terms for the velocities OR a reduced
    basis summarizing degenerate terms that appear only together for
    the redshift-space P(k).
    [This is documented in Equation 5.1 of Chen, Vlah and White (2020).]
    
    Note that in practice some parameters, such as b3, might be set
    additionally to zero to restrict parameter space.
    We leave this up to the user.
    '''

    def __init__(self, *args, kmin = 5e-3, kmax = 0.3, nk = 50, **kw):
        '''
        Same keywords and arguments as the other two classes for now.
        '''
        
        # Setup ffts etc.
        VelocityMoments.__init__(self, *args, **kw)

        self.kmin, self.kmax, self.nk = kmin, kmax, nk
        self.kv = np.logspace(np.log10(kmin), np.log10(kmax), nk); self.nk = nk
        self.weight = self.kv**3 / (2*np.pi**2)
        
        self.setup_velocity_moments()

    def setup_velocity_moments(self):
        self.make_ptable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_vtable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_spartable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_stracetable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.convert_sigma_bases(basis='Polynomial')
        
        if self.beyond_gauss:
            self.make_gamma1table(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
            self.make_gamma2table(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
            self.convert_gamma_bases()

            self.make_kappatable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
            self.convert_kappa_bases()
        

    def compute_redshift_space_power_at_mu(self,pars,f,mu_obs,counterterm_c3=0,reduced=False,apar=1,aperp=1):
        '''
        Moment expansion approach.
        
        The "reduced" basis of stochastic and counterterms is equivalent
        to Equation 5.1 in Chen, Vlah & White (2020).
        
        If AP parameters apar and aperp are nonzero then the input/output  k, mu refer to the observed.
        We use "physical" AP parameters, defined as the scaling of distances parallel and perpendicular
        to the line of sight.
        
        '''
        # Change mu to the "true" from the input observed
        # Note that kv below refers to "true" k
        # We follow the notation/conventions in https://arxiv.org/abs/1312.4611
        # Eqs. (58-60).
        F = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F**2 - 1) )
        mu = mu_obs / F / AP_fac
        
        # If using a reduced vector, make a new one.

        mu2 = mu**2
        if self.beyond_gauss:
            if reduced:
                b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
                
                kv, pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha0,sn)
                kv, vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alpha2,sn2)
                kv, s0, s2 = self.combine_bias_terms_sk(b1,b2,bs,b3,0,alpha4,0,basis='Polynomial')
                kv, g1, g3 = self.combine_bias_terms_gk(b1,b2,bs,b3,0,alpha6)
                kv, k0, k2, k4 = self.combine_bias_terms_kk(0,sn4)
                
            else:
                b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, alpha_g1, alpha_g3, alpha_k2, sn, sv, sigma0_stoch, sn4= pars
                
                kv, pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha,sn)
                kv, vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alpha_v,sv)
                kv, s0, s2 = self.combine_bias_terms_sk(b1,b2,bs,b3,alpha_s0,alpha_s2,sigma0_stoch,basis='Polynomial')
                kv, g1, g3 = self.combine_bias_terms_gk(b1,b2,bs,b3,alpha_g1,alpha_g3)
                kv, k0, k2, k4 = self.combine_bias_terms_kk(alpha_k2,sn4)
                
            ret = pk - f * kv * mu2 * vk -\
                  0.5 * f**2 * kv**2 * mu2 * ( s0 + s2* mu2 ) +\
                  1./6 * f**3 * kv**3 * mu * (g1 + mu2 * g3) +\
                  1./24 * f**4 * kv**4 * (k0 + mu2*k2 + mu2**2*k4)
                  
        else:
            if reduced:
                b1, b2, bs, b3, alpha0, alpha2, alpha4, sn, sn2 = pars
                bv = [b1, b2, bs, b3, alpha0, alpha2, 0, 0, sn, 0, sn2 ]
                ct3 = alpha4
                
                kv, pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha0,sn)
                kv, vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alpha2,sn2)
                kv, s0, s2 = self.combine_bias_terms_sk(b1,b2,bs,b3,0,0,0,basis='Polynomial')
                
            else:
                b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, sigma0_stoch = pars
                ct3 = counterterm_c3
                
                kv, pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha,sn)
                kv, vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alpha_v,sv)
                kv, s0, s2 = self.combine_bias_terms_sk(b1,b2,bs,b3,alpha_s0,alpha_s2,sigma0_stoch,basis='Polynomial')

            mu2 = mu**2
            ret = pk - f * kv * mu2 * vk -\
                0.5 * f**2 * kv**2 * mu2 * ( s0 + s2* mu2 ) +\
                ct3 /6. * self.kv**2 * mu2**2 * self.pktable[:,-1]
        
        # Interpolate onto true wavenumbers
        kobs = self.kv * aperp / AP_fac
        pks_obs = interp1d(kobs, ret, kind='cubic', fill_value='extrapolate')(self.kv)
        pks_obs = pks_obs / aperp**2 / apar
        #pks_obs = np.interp(self.kv, kobs, ret)
        
        return self.kv, pks_obs


    def compute_redshift_space_power_multipoles(self, pars, f, counterterm_c3=0, ngauss=4, reduced=False, apar=1, aperp = 1):

        # Generate the sampling
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),self.nk))
        
        for ii, nu in enumerate(nus_calc):
            self.pknutable[ii,:] = self.compute_redshift_space_power_at_mu(pars,f,nu,reduced=reduced,counterterm_c3=counterterm_c3,apar=apar,aperp=aperp)[1]
                
        self.pknutable[ngauss:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None]*self.pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None]*self.pknutable,axis=0)
        
        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable

