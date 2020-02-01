import numpy as np

from velocity_moments_fftw import VelocityMoments

class FourierStreamingModel(VelocityMoments):
    '''
    Class to calculate the redshift space power spectrum in the Fourier streaming model.
    
    Inherits the VelocityMoments class which itself inherits the CLEFT class.
    
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

    
    
    def compute_cumulants(self, bvec):
        '''
        Calculate velocity moments and turn into cumulants.
        The bvec format is [b1, b2, bs, alpha, alpha_v, alpha_s0, alpha_s2, sn, sv, s0]
        
        '''

        # Compute each moment
        kv, pk = self.combine_bias_terms_pk(bvec)
        kv, vk = self.combine_bias_terms_vk(bvec)
        kv, s0, s2 = self.combine_bias_terms_sk(bvec,basis='Legendre')
        
        # Turn into cumulants; split into coefficients of LOS angle nu
        self.one_plus_delta = 1 + self.weight * pk

        # first cumulant
        # this is imaginary and goes into the expansion as exp(- (k*nu) im_c1 * nu ...)
        self.im_c1 = self.weight * vk / self.one_plus_delta

        # second cumulant: note that sigma_zz = sigma0 + sigma2 * 3/2 (mu^2 - 1/3) = (sigma0 -1/2 sigma 2) + 3/2 * sigma2 * mu^2
        self.c2_0 = self.weight * (s0 - 0.5*s2)/self.one_plus_delta
        self.c2_2 = self.weight * (1.5*s2)/self.one_plus_delta + self.im_c1**2

    def compute_redshift_space_power(self, bvec, f, nu, counterterm_c3=0):

        self.compute_cumulants(bvec)
        
        nu2 = nu**2
        expon = -f * self.kv * nu2 * self.im_c1 - 0.5 * f**2 * self.kv**2 * nu2 * (self.c2_0 + self.c2_2*nu2)
        expon += counterterm_c3/6. * self.weight * self.kv**2 * nu**4 * self.pktable[:,1] / self.one_plus_delta

        return self.kv, (self.one_plus_delta * np.exp(expon) - 1.)/self.weight


    def compute_redshift_space_power_multipoles(self, bvec, f, counterterm_c3=0, ngauss=4, method='FSM'):

        # Generate the sampling
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),self.nk))
        
        for ii, nu in enumerate(nus_calc):
            if method == 'FSM':
                self.pknutable[ii,:] = self.compute_redshift_space_power(bvec,f,nu,counterterm_c3=counterterm_c3)[1]
            elif method == 'Moments':
                self.pknutable[ii,:] = self.compute_redshift_space_power_moments(bvec,f,nu)[1]
                
        self.pknutable[ngauss:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None]*self.pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None]*self.pknutable,axis=0)
        
        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable

