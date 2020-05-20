import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import tukey

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

    def __init__(self, *args, kmin=1e-3, kmax=3, nk= 200, jn = 10, cutoff=20, kswitch=3e-3, **kw):
        '''
        Same keywords and arguments as the other two classes for now.
        '''
        # Setup ffts etc.
        VelocityMoments.__init__(self, *args, **kw)

        self.kmin, self.kmax, self.nk = kmin, kmax, nk
        self.kv = np.logspace(np.log10(kmin), np.log10(kmax), nk); self.nk = nk
        
        self.kint = np.logspace(-5,3,4000)
        self.plin = loginterp(self.k, self.p)(self.kint)
        self.sph_gsm  = SphericalBesselTransform(self.kint,L=3,fourier=True)
        self.rint = np.logspace(-3,5,4000)
        self.rint = self.rint[(self.rint>0.1)*(self.rint<600)] #actual range of integration
        
        self.window = tukey(4000)
        self.weight =  0.5 * (1 + np.tanh(3*np.log(self.kint/kswitch)))
        self.peft = None
        self.veft = None
        self.s0eft = None
        self.s2eft = None
        
        self.setup_velocity_moments()
        #self.setup_config_vels()

    def setup_velocity_moments(self):
        self.make_ptable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_vtable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_spartable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.make_stracetable(kmin = self.kmin, kmax = self.kmax, nk = self.nk)
        self.convert_sigma_bases(basis='Legendre')

    
    def compute_cumulants(self, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog):
        '''
        Calculate velocity moments and turn into cumulants.
        '''
        # Compute each moment
        
        self.kv   = self.pktable[:,0]
        self.pzel = self.pktable[:,-1]
        
        self.peft = self.pktable[:,1] + b1*self.pktable[:,2] + b1**2*self.pktable[:,3]\
        + b2*self.pktable[:,4] + b1*b2*self.pktable[:,5] + b2**2 * self.pktable[:,6]\
        + bs*self.pktable[:,7] + b1*bs*self.pktable[:,8] + b2*bs*self.pktable[:,9]\
        + bs**2*self.pktable[:,10] + b3*self.pktable[:,11] + b1*b3*self.pktable[:,12] + alpha* self.kv**2 * self.pzel
        
        _integrand = loginterp(self.kv, self.peft)(self.kint) #at some point turn into theory extrapolation
        _integrand = self.weight * _integrand + (1-self.weight) * ((1+b1)**2 * self.plin)
        
        qint, xi = self.sph_gsm.sph(0,_integrand)
        self.xieft = np.interp(self.rint, qint, xi)
                
        self.vkeft = self.vktable[:,1] + b1*self.vktable[:,2] + b1**2*self.vktable[:,3]\
        + b2*self.vktable[:,4] + b1*b2*self.vktable[:,5] \
        + bs*self.vktable[:,7] + b1*bs*self.vktable[:,8] + b3 * self.vktable[:,11]\
        + alpha_v * self.kv * self.pzel
        
        _integrand = loginterp(self.kv, self.vkeft)(self.kint)
        _integrand = self.weight * _integrand + (1-self.weight) * (-2 * (1+b1) * self.plin/self.kint)

        qint, xi = self.sph_gsm.sph(1,_integrand)
        self.veft = np.interp(self.rint, qint, xi)
        
        self.s0keft =  self.s0[:,1] + b1*self.s0[:,2] + b1**2*self.s0[:,3]\
                                                           + b2*self.s0[:,4] \
                                                           + bs*self.s0[:,7] \
                                                           + alpha_s0 * self.pzel
        
        self.s2keft =  self.s2[:,1] + b1*self.s2[:,2] + b1**2*self.s2[:,3]\
                                                           + b2*self.s2[:,4] \
                                                           + bs*self.s2[:,7] \
                                                           + alpha_s2 * self.pzel
        
        _integrand = loginterp(self.kv, self.s0keft)(self.kint)
        _integrand = self.weight * _integrand + (1-self.weight) * (-2./3 * self.plin/self.kint**2)

        qint, xi = self.sph_gsm.sph(0,_integrand * self.window)
        self.s0eft = np.interp(self.rint, qint, xi)

        _integrand = loginterp(self.kv, self.s2keft)(self.kint)
        _integrand = self.weight * _integrand + (1-self.weight) * (-4./3 * self.plin/self.kint**2)

        qint2, xi = self.sph_gsm.sph(2,_integrand * self.window); xi *=-1
        self.s2eft = np.interp(self.rint, qint2, xi)
                                                           
        self.s0eft += (self.Xddot + self.Xloopddot + 2*b1*self.X10ddot + 2*bs*self.Xs2ddot)[-1] + s2fog #add in 0-lag term
                                                           



    def compute_xi_rsd(self, sperp, spar, f, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog, rwidth=100, Nint=10000, update_cumulants=False):
        '''
        Compute the redshift-space xi(sperpendicular,sparallel).
        '''
        # If cumulants have already been computed, skip this step:
        if update_cumulants or (self.peft is None):
            print("Here.")
            self.compute_cumulants(b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog)

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


    def compute_xi_ell(self, s, f, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog, rwidth=100, Nint=10000, ngauss=4, update_cumulants=False):
        '''
        Compute the redshift-space correlation function multipoles
        '''
        # Compute the cumulants
        #self.compute_cumulants(b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog)
        
        # Compute each moment
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
            
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        nus_calc = nus[:ngauss]
        
        xi0, xi2, xi4 = 0,0,0
        for ii, nu in enumerate(nus_calc):
            xi_nu = self.compute_xi_rsd(s*np.sqrt(1-nu**2),s*nu, f, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog, rwidth=rwidth, Nint=Nint, update_cumulants=False)
            xi0 += xi_nu * L0[ii] * 1 * ws[ii]
            xi2 += xi_nu * L2[ii] * 5 * ws[ii]
            xi4 += xi_nu * L4[ii] * 9 * ws[ii]
        return xi0, xi2, xi4


    def compute_xi_real(self, rr, b1, b2, bs, b3, alpha, alpha_v, alpha_s0, alpha_s2, s2fog):
        '''
        Compute the real-space correlation function at rr.
        '''
        # This is just the zeroth moment:
        self.kv   = self.pktable[:,0]
        self.pzel = self.pktable[:,-1]
        
        self.peft = self.pktable[:,1] + b1*self.pktable[:,2] + b1**2*self.pktable[:,3]\
        + b2*self.pktable[:,4] + b1*b2*self.pktable[:,5] + b2**2 * self.pktable[:,6]\
        + bs*self.pktable[:,7] + b1*bs*self.pktable[:,8] + b2*bs*self.pktable[:,9]\
        + bs**2*self.pktable[:,10] + b3*self.pktable[:,11] + b1*b3*self.pktable[:,12] + alpha* self.kv**2 * self.pzel
        
        _integrand = loginterp(self.kv, self.peft)(self.kint) #at some point turn into theory extrapolation
        _integrand = self.weight * _integrand + (1-self.weight) * ((1+b1)**2 * self.plin)
        
        qint, xi = self.sph_gsm.sph(0,_integrand)
        
        xir = Spline(qint,xi)(rr)
        return xir
