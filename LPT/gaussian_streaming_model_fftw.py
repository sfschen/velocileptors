import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from Utils.spherical_bessel_transform import SphericalBesselTransform
from Utils.loginterp import loginterp


from LPT.velocity_moments_fftw import VelocityMoments


class GaussianStreamingModel(VelocityMoments):
    '''
    Class to calculate the redshift space correlation function
    using the Gaussian streaming model.
    
    Inherits the VelocityMoments class which itself inherits the CLEFT class.
    
    Note if third_order = False passing b_3 to the functions will simply produce no effect.
    '''

    def __init__(self, *args, kmin=1e-3, kmax=3, nk= 200, jn = 10, cutoff=20, **kw):
        '''
        Same keywords and arguments as the other two classes for now.
        '''
        # Setup ffts etc.
        VelocityMoments.__init__(self, *args, **kw)

        self.kmin, self.kmax, self.nk = kmin, kmax, nk
        self.kv = np.logspace(np.log10(kmin), np.log10(kmax), nk); self.nk = nk
        
        self.kint = np.logspace(-5,3,4000)
        self.sph_gsm  = SphericalBesselTransform(self.kint,L=3,fourier=True)
        self.rint = np.logspace(-3,5,4000)
        self.rint = self.rint[(self.rint>0.1)*(self.rint<600)] #actual range of integration
        
        self.setup_velocity_moments()
        self.setup_config_vels()

    def setup_velocity_moments(self):
        self.make_ptable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_vtable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_spartable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_stracetable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.convert_sigma_bases(basis='Legendre')

    def setup_config_vels(self):
        # Fourier transform the velocity moments
        # the correlation function
        self.xitable = np.zeros((len(self.rint),13)) # minus one because we deal with the ct separately

        for ii in range(self.num_power_components-1):
            _integrand = loginterp(self.pktable[:,0], self.pktable[:,1+ii])(self.kint)
            qs, xs = self.sph_gsm.sph(0,_integrand)
            self.xitable[:,ii] = np.interp(self.rint, qs, xs)
            
        _integrand = loginterp(self.pktable[:,0], self.pktable[:,1])(self.kint)
        qint, ximatter = self.sph_gsm.sph(0,_integrand)
        self.ximatter = np.interp(self.rint, qint, ximatter)

        _integrand = loginterp(self.pktable[:,0], self.pktable[:,0]**2 * self.pktable[:,-1])(self.kint)
        qint, xict = self.sph_gsm.sph(0,_integrand)
        self.xict = np.interp(self.rint, qint, xict)
        
        # the pairwise velocity
        self.vtable = np.zeros((len(self.rint),12))

        for ii in range(self.num_power_components-1):
            _integrand = loginterp(self.vktable[:,0], self.vktable[:,1+ii])(self.kint)
            qs, xs = self.sph_gsm.sph(1,_integrand)
            self.vtable[:,ii] = np.interp(self.rint, qs, xs)
            
        _integrand = loginterp(self.vktable[:,0], self.vktable[:,1])(self.kint)
        qint, ximatter = self.sph_gsm.sph(1,_integrand)
        self.vmatter = np.interp(self.rint, qint, ximatter)

        _integrand = loginterp(self.vktable[:,0], self.vktable[:,0] * self.pktable[:,-1])(self.kint)
        qint, xict = self.sph_gsm.sph(1,_integrand)
        self.vct = np.interp(self.rint, qint, xict)
        
        # and finally the velocity dispersions
        self.s0table = np.zeros((len(self.rint),12))
        self.s2table = np.zeros((len(self.rint),12))
        for ii in range(self.num_power_components-1):
            _integrand = loginterp(self.s0[1:,0], self.s0[1:,1+ii])(self.kint)
            qs, xs = self.sph_gsm.sph(0,_integrand)
            self.s0table[:,ii] = np.interp(self.rint, qs, xs)
            
            _integrand = loginterp(self.s2[1:,0], self.s2[1:,1+ii])(self.kint)
            qs, xs  = self.sph_gsm.sph(2,_integrand)
            self.s2table[:,ii] = np.interp(self.rint, qs, xs)
            
        self.s2table *= -1
            
        _integrand = loginterp(self.sparktable[10:,0], self.s0[10:,1])(self.kint)
        qint, s0matter = self.sph_gsm.sph(0,_integrand)
        self.s0matter = np.interp(self.rint, qint, s0matter)

        _integrand = loginterp(self.sparktable[5:,0], self.s2[5:,1])(self.kint)
        qint2, s2matter = self.sph_gsm.sph(2,_integrand); s2matter *=-1
        self.s2matter = np.interp(self.rint, qint2, s2matter)

        _integrand = loginterp(self.pktable[:,0], self.pktable[:,-1])(self.kint)
        qint, xict = self.sph_gsm.sph(0,_integrand)
        self.s0ct = np.interp(self.rint, qint, xict)

        _integrand = loginterp(self.pktable[:,0],self.pktable[:,0]**0 * self.pktable[:,-1])(self.kint)
        qint2, s2ct = self.sph_gsm.sph(2,_integrand); s2ct *=-1
        self.s2ct = np.interp(self.rint, qint2, s2ct)
        
        
        
    
    def compute_cumulants(self, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog):
        '''
        Calculate velocity moments and turn into cumulants.
        '''
        # Compute each moment
        self.xieft = self.ximatter + b1*self.xitable[:,1] + b1**2*self.xitable[:,2]\
        + b2*self.xitable[:,3] + b1*b2*self.xitable[:,4] + b2**2 * self.xitable[:,5]\
        + bs*self.xitable[:,6] + b1*bs*self.xitable[:,7] + b2*bs*self.xitable[:,8]\
        + bs**2*self.xitable[:,9] + b3*self.xitable[:,10] + b1*b3*self.xitable[:,11] + alpha*self.xict
        
        self.veft = self.vmatter + b1*self.vtable[:,1] + b1**2*self.vtable[:,2]\
        + b2*self.vtable[:,3] + b1*b2*self.vtable[:,4] \
        + bs*self.vtable[:,6] + b1*bs*self.vtable[:,7] + b3 * self.vtable[:,10]\
        + alpha_v*self.vct
        
        self.s0eft =  self.s0matter + b1*self.s0table[:,1] + b1**2*self.s0table[:,2]\
                                                           + b2*self.s0table[:,3] \
                                                           + bs*self.s0table[:,6] \
                                                           + alpha_s0 * self.s0ct \
                                                           + s2fog
        self.s0eft += (self.Xddot + self.Xloopddot + 2*b1*self.X10ddot + 2*bs*self.Xs2ddot)[-1] #add in 0-lag term
                                                           
        self.s2eft =  self.s2matter + b1*self.s2table[:,1] + b1**2*self.s2table[:,2]\
                                                           + b2*self.s2table[:,3] \
                                                           + bs*self.s2table[:,6] \
                                                           + alpha_s2 * self.s2ct


    def compute_xi_rsd(self, sperp_obs, spar_obs, f, b1, b2, bs, b3, alpha, alpha_v, s2fog, alpha_s0, alpha_s2, apar=1.0, aperp=1.0, rwidth=100, Nint=10000, update_cumulants=True):
        '''
        Compute the redshift-space xi(sperpendicular,sparallel).
        '''
        # If cumulants have already been computed, skip this step:
        if update_cumulants:
            self.compute_cumulants(b1, b2, bs, b3, alpha, alpha_v, s2fog, alpha_s0, alpha_s2)

        # define "true" coordinates using A-P parameters.
        spar  = spar_obs  * apar
        sperp = sperp_obs * aperp

        # definte integration coords
        ys = np.linspace(-rwidth,rwidth,Nint) # this z - s_par
        rs = np.sqrt( (spar - ys)**2 + sperp**2 )
        mus = (spar - ys)/rs
        
        xi_int = 1 + np.interp(rs, self.rint, self.xieft)
        v_int  = f*( np.interp(rs, self.rint, self.veft) * mus ) / xi_int
        s_int  = f**2 * ( np.interp(rs, self.rint, self.s0eft) + 0.5 * (3*mus**2 - 1) * np.interp(rs, self.rint, self.s2eft) )/xi_int - v_int**2

        
        integrand = xi_int * np.exp( -0.5 * (ys - v_int)**2 / s_int ) / np.sqrt(2*np.pi*s_int)
        integrand[np.isnan(integrand)] = 0.
        return np.trapz(integrand, x=ys) - 1


    def compute_xi_ell(self, s, f, b1, b2, bs, b3, alpha, alpha_v, s2fog, alpha_s0, alpha_s2, apar=1.0, aperp=1.0,  rwidth=100, Nint=10000, ngauss=4):
        '''
        Compute the redshift-space correlation function multipoles
        '''
        # Compute the cumulants
        self.compute_cumulants(b1, b2, bs, b3, alpha, alpha_v, s2fog, alpha_s0, alpha_s2)
        
        # Compute each moment
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
            
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        nus_calc = nus[:ngauss]
        
        xi0, xi2, xi4 = 0,0,0
        for ii, nu in enumerate(nus_calc):
            xi_nu = self.compute_xi_rsd(s*np.sqrt(1-nu**2),s*nu, f, b1, b2, bs, b3, alpha, alpha_v, s2fog, alpha_s0, alpha_s2, apar=apar, aperp=aperp, rwidth=rwidth, Nint=Nint, update_cumulants=False)
            xi0 += xi_nu * L0[ii] * 1 * ws[ii]
            xi2 += xi_nu * L2[ii] * 5 * ws[ii]
            xi4 += xi_nu * L4[ii] * 9 * ws[ii]
        return xi0, xi2, xi4


    def compute_xi_real(self, rr, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog):
        '''
        Compute the real-space correlation function at rr.
        '''
        # This is just the zeroth moment:
        xieft = self.ximatter + b1*self.xitable[:,1] + b1**2*self.xitable[:,2]\
        + b2*self.xitable[:,3] + b1*b2*self.xitable[:,4]\
        + b2**2 * self.xitable[:,5]\
        + bs*self.xitable[:,6] + b1*bs*self.xitable[:,7]\
        + b2*bs*self.xitable[:,8]\
        + bs**2*self.xitable[:,9] + b3*self.xitable[:,10]\
        + b1*b3*self.xitable[:,11] + alpha*self.xict
        xir = Spline(self.rint,xieft)(rr)
        return xir
