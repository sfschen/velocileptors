import numpy as np

from velocileptors.LPT.velocity_moments_fftw import VelocityMoments

class FourierStreamingModel(VelocityMoments):
    '''
    Class to calculate the redshift space power spectrum in the Fourier streaming model.
    
    Inherits the VelocityMoments class which itself inherits the CLEFT class.
    
    Here we compute up to the first two moments v(k) and sigma(k), in addition
    to a counterterm ansatz for the third moment, which is an excellent approximation to the data.
    
    The full one-loop expression can be computed in the moment expansion class if desired,
    but we note that the inclusion of higher moments did not noticeably improve agreement
    with data in n-body simulations.
    
    '''

    def __init__(self, *args, kmin = 1e-3, kmax = 3, nk = 100, **kw):
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
        self.convert_sigma_bases(basis='Legendre')

    
    
    def compute_cumulants(self, pars):
        '''
        Calculate velocity moments and turn into cumulants.
        The pars format is [b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, sigma0_stoch]
        
        '''
        # Compute each moment
        b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, sigma0_stoch = pars
        
        kv, pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha,sn)
        kv, vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alpha_v,sv)
        kv, s0, s2 = self.combine_bias_terms_sk(b1,b2,bs,b3,alpha_s0,alpha_s2,sigma0_stoch,basis='Legendre')

        # Turn into cumulants; split into coefficients of LOS angle mu
        self.one_plus_delta = 1 + self.weight * pk

        # first cumulant
        # this is imaginary and goes into the expansion as exp(- (k*mu) im_c1 * mu ...)
        self.im_c1 = self.weight * vk / self.one_plus_delta

        # second cumulant: note that sigma_zz = sigma0 + sigma2 * 3/2 (mu^2 - 1/3) = (sigma0 -1/2 sigma 2) + 3/2 * sigma2 * mu^2
        self.c2_0 = self.weight * (s0 - 0.5*s2)/self.one_plus_delta
        self.c2_2 = self.weight * (1.5*s2)/self.one_plus_delta + self.im_c1**2

    def compute_redshift_space_power_at_mu(self, pars, f, mu, counterterm_c3=0):

        self.compute_cumulants(pars)
        
        mu2 = mu**2
        expon = -f * self.kv * mu2 * self.im_c1 - 0.5 * f**2 * self.kv**2 * mu2 * (self.c2_0 + self.c2_2*mu2)
        expon += counterterm_c3/6. * self.weight * self.kv**2 * mu**4 * self.pktable[:,-1] / self.one_plus_delta

        return self.kv, (self.one_plus_delta * np.exp(expon) - 1.)/self.weight


    def compute_redshift_space_power_multipoles(self, pars, f, counterterm_c3=0, ngauss=4):

        # Generate the sampling
        mus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        mus_calc = mus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(mus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(mus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(mus)
        
        self.pkmutable = np.zeros((len(mus),self.nk))
        
        for ii, mu in enumerate(mus_calc):
            self.pkmutable[ii,:] = self.compute_redshift_space_power_at_mu(pars,f,mu,counterterm_c3=counterterm_c3)[1]

                
        self.pkmutable[ngauss:,:] = np.flip(self.pkmutable[0:ngauss],axis=0)
        
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None]*self.pkmutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None]*self.pkmutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None]*self.pkmutable,axis=0)
        
        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable

