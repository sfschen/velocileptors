import numpy as np

from velocity_moments_fftw import VelocityMoments

class MomentExpansion(VelocityMoments):
    '''
    Class to calculate the redshift space power spectrum in the moment expansion approach.
    
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
        

    def compute_redshift_space_power_at_mu(self, bvec, f, nu, counterterm_c3=0, reduced=False):
        '''
        Moment expansion approach.
        
        The "reduced" basis of stochastic and counterterms is equivalent to Equation 5.1 in the paper.
        
        '''
        # If using a reduced vector, make a new one.
        if reduced:
            b1, b2, bs, alpha0, alpha2, alpha4, sn, sn2 = bvec
            bv = [b1, b2, bs, alpha0, alpha2, 0, 0, sn, 0, sn2 ]
            ct3 = alpha4
        else:
            bv = bvec
            ct3 = counterterm_c3
        
        
        # Compute each moment
        kv, pk = self.combine_bias_terms_pk(bv)
        kv, vk = self.combine_bias_terms_vk(bv)
        kv, s0, s2 = self.combine_bias_terms_sk(bv,basis='Polynomial')

        nu2 = nu**2
        ret = pk - f * kv * nu2 * vk - 0.5 * f**2 * kv**2 * nu2 * ( s0 + s2* nu2 )
        ret += ct3 /6. * self.kv**2 * nu2**2 * self.pktable[:,1]

        return self.kv, ret


    def compute_redshift_space_power_multipoles(self, bvec, f, counterterm_c3=0, ngauss=4, reduced=False):

        # Generate the sampling
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),self.nk))
        
        for ii, nu in enumerate(nus_calc):
            self.pknutable[ii,:] = self.compute_redshift_space_power_at_mu(bvec,f,nu,reduced=reduced,counterterm_c3=counterterm_c3)[1]
                
        self.pknutable[ngauss:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None]*self.pknutable,axis=0)
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None]*self.pknutable,axis=0)
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None]*self.pknutable,axis=0)
        
        return self.kv, self.p0ktable, self.p2ktable, self.p4ktable

