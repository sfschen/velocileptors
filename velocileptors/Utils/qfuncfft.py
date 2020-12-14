import numpy as np

from Utils.loginterp import loginterp
from Utils.spherical_bessel_transform import SphericalBesselTransform



class QFuncFFT:
    '''
       Class to calculate all the functions of q, X(q), Y(q), U(q), xi(q) etc.
       as well as the one-loop terms Q_n(k), R_n(k) using FFTLog.
       
       Throughout we use the ``generalized correlation function'' notation of 1603.04405.
       
       This is modified to take an IR scale kIR
              
       Note that one should always cut off the input power spectrum above some scale.
       I use exp(- (k/20)^2 ) but a cutoff at scales twice smaller works equivalently,
       and probably beyond that. The important thing is to keep all integrals finite.
       This is done automatically in the Zeldovich class.
       
       Currently using the numpy version of fft. The FFTW takes longer to start up and
       the resulting speedup is unnecessary in this case.
       
    '''
    def __init__(self, k, p, kIR = None, qv = None, oneloop = False, shear = True, third_order = True, low_ring=True):

        self.oneloop = oneloop
        self.shear = shear
        self.third_order = third_order
        
        self.k = k
        self.p = p
        
        if kIR is not None:
            self.ir_less = np.exp(- (self.k/kIR)**2 )
            self.ir_greater = -np.expm1(- (self.k/kIR)**2)
        else:
            self.ir_less = 1
            self.ir_greater = 0

        if qv is None:
            self.qv = np.logspace(-5,5,2e4)
        else:
            self.qv = qv
        
        self.sph = SphericalBesselTransform(self.k, L=5, low_ring=True, fourier=True)
        self.sphr = SphericalBesselTransform(self.qv, L=5, low_ring=True, fourier=False)

        
        self.setup_xiln()
        self.setup_2pts()
        
        if self.shear:
            self.setup_shear()
        
        if self.oneloop:
            self.setup_QR()
            self.setup_oneloop_2pts()
            
        if self.third_order:
            self.setup_third_order()

    def setup_xiln(self):
        
        # Compute a bunch of generalized correlation functions
        self.xi00 = self.xi_l_n(0,0)
        self.xi1m1 = self.xi_l_n(1,-1)
        self.xi0m2 = self.xi_l_n(0,-2, side='right') # since this approaches constant on the left only interpolate on right
        self.xi2m2 = self.xi_l_n(2,-2)
        
        # Also compute the IR-cut lm2's
        self.xi0m2_lt = self.xi_l_n(0,-2, IR_cut = 'lt', side='right')
        self.xi2m2_lt = self.xi_l_n(2,-2, IR_cut = 'lt')
        
        #self.xi0m2_gt = self.xi_l_n(0,-2, IR_cut = 'gt', side='right')
        #self.xi2m2_gt = self.xi_l_n(2,-2, IR_cut = 'gt')
    
        # also compute those for one loop terms since they don't take much more time
        # also useful in shear terms
        self.xi20 = self.xi_l_n(2,0)
        self.xi40 = self.xi_l_n(4,0)
        
        self.xi11 = self.xi_l_n(1,1)
        self.xi31 = self.xi_l_n(3,1)
        self.xi3m1 = self.xi_l_n(3,-1)
        
        self.xi02 = self.xi_l_n(0,2)
        self.xi22 = self.xi_l_n(2,2)
    
    def setup_QR(self):
    
        # Computes Q_i(k), R_i(k)-- technically will want them transformed again!

        # then lump together into the kernels and reverse fourier
        Qfac = 4 * np.pi
        _integrand_Q1 = Qfac * (8./15 * self.xi00**2 - 16./21 * self.xi20**2 + 8./35 * self.xi40**2)
        _integrand_Q2 = Qfac * (4./5 * self.xi00**2 - 4./7 * self.xi20**2 - 8./35 * self.xi40**2 \
                                - 4./5 * self.xi11*self.xi1m1 + 4/5 * self.xi31*self.xi3m1)
        _integrand_Q3 = Qfac * (38./15 * self.xi00**2 + 2./3*self.xi02*self.xi0m2 \
                                - 32./5*self.xi1m1*self.xi11 + 68./21*self.xi20**2 \
                                + 4./3 * self.xi22*self.xi2m2 - 8./5 * self.xi31*self.xi3m1 + 8./35*self.xi40**2)
        _integrand_Q5 = Qfac * (2./3 * self.xi00**2 - 2./3*self.xi20**2 \
                                - 2./5 * self.xi11*self.xi1m1 + 2./5 * self.xi31*self.xi3m1)
        _integrand_Q8 = Qfac * (2./3 * self.xi00**2 - 2./3*self.xi20**2)
        _integrand_Qs2 = Qfac * (-4./15 * self.xi00**2 + 20./21*self.xi20**2 - 24./35*self.xi40**2)
                                
        self.Q1 = self.template_QR(0, _integrand_Q1)
        self.Q2 = self.template_QR(0, _integrand_Q2)
        self.Q3 = self.template_QR(0, _integrand_Q3)
        self.Q5 = self.template_QR(0, _integrand_Q5)
        self.Q8 = self.template_QR(0, _integrand_Q8)
        self.Qs2 = self.template_QR(0, _integrand_Qs2)
    
        _integrand_R1_0 = self.xi00/self.qv
        _integrand_R1_2 = self.xi20/self.qv
        _integrand_R1_4 = self.xi40/self.qv
        _integrand_R2_1 = self.xi1m1/self.qv
        _integrand_R2_3 = self.xi3m1/self.qv

        R1_0 = self.template_QR(0,_integrand_R1_0)
        R1_2 = self.template_QR(2,_integrand_R1_2)
        R1_4 = self.template_QR(4,_integrand_R1_4)
        R2_1 = self.template_QR(1,_integrand_R2_1)
        R2_3 = self.template_QR(3,_integrand_R2_3)

        self.R1 = self.k**2 * self.p * (8./15 * R1_0 - 16./21* R1_2 + 8./35 * R1_4)
        self.R2 = self.k**2 *self.p * (-2./15 * R1_0 - 2./21* R1_2 + 8./35 * R1_4 +  self.k * 2./5*R2_1 - self.k* 2./5*R2_3)

    def setup_2pts(self):
        # Piece together xi_l_n into what we need
        self.Xlin = 2./3 * (self.xi0m2[0] - self.xi0m2 - self.xi2m2)
        self.Ylin = 2 * self.xi2m2
        
        self.Xlin_lt = 2./3 * (self.xi0m2_lt[0] - self.xi0m2_lt - self.xi2m2_lt)
        self.Ylin_lt = 2 * self.xi2m2_lt
        
        self.Xlin_gt = self.Xlin - self.Xlin_lt
        self.Ylin_gt = self.Ylin - self.Ylin_lt
        
        #self.Xlin_gt = 2./3 * (self.xi0m2_gt[0] - self.xi0m2_gt - self.xi2m2_gt)
        #self.Ylin_gt = 2 * self.xi2m2_gt
        
        self.Ulin = - self.xi1m1
        self.corlin = self.xi00
    
    def setup_shear(self):
        # Let's make some (disconnected) shear contributions
        J2 = 2.*self.xi1m1/15 - 0.2*self.xi3m1
        J3 = -0.2*self.xi1m1 - 0.2*self.xi3m1
        J4 = self.xi3m1
        
        self.V = 4 * J2 * self.xi20
        self.Xs2 = 4 * J3**2
        self.Ys2 = 6*J2**2 + 8*J2*J3 + 4*J2*J4 + 4*J3**2 + 8*J3*J4 + 2*J4**2
        self.zeta = 2*(4*self.xi00**2/45. + 8*self.xi20**2/63. + 8*self.xi40**2/35)
        self.chi  = 4*self.xi20**2/3.
    
    def setup_oneloop_2pts(self):
        # same as above but for all the one loop pieces
        
        # Aij 1 loop
        self.xi0m2loop13 = self.xi_l_n(0,-2, _int=5./21*self.R1)
        self.xi2m2loop13 = self.xi_l_n(2,-2, _int=5./21*self.R1)
        
        self.Xloop13 = 2./3 * (self.xi0m2loop13[0] - self.xi0m2loop13 - self.xi2m2loop13)
        self.Yloop13 = 2 * self.xi2m2loop13
        
        self.xi0m2loop22 = self.xi_l_n(0,-2, _int=9./98*self.Q1)
        self.xi2m2loop22 = self.xi_l_n(2,-2, _int=9./98*self.Q1)

        self.Xloop22 = 2./3 * (self.xi0m2loop22[0] - self.xi0m2loop22 - self.xi2m2loop22)
        self.Yloop22 = 2 * self.xi2m2loop22
        
        # Wijk
        self.Tloop112 = self.xi_l_n(3,-3, _int=-3./7*(2*self.R1+4*self.R2+self.Q1+2*self.Q2))
        self.V1loop112 = self.xi_l_n(1,-3,_int=3./35*(-3*self.R1+4*self.R2+self.Q1+2*self.Q2)) - 0.2*self.Tloop112
        self.V3loop112 = self.xi_l_n(1,-3,_int=3./35*(2*self.R1+4*self.R2-4*self.Q1+2*self.Q2)) - 0.2*self.Tloop112
        
        # A10
        self.zerolag_10_loop12 = np.trapz((self.R1-self.R2)/7.,x=self.k) / (2*np.pi**2)
        self.xi0m2_10_loop12 = self.xi_l_n(0,-2, _int=4*self.R2+2*self.Q5)/14.
        self.xi2m2_10_loop12 = self.xi_l_n(2,-2, _int=3*self.R1+4*self.R2+2*self.Q5)/14.
        
        self.X10loop12 = self.zerolag_10_loop12 - self.xi0m2_10_loop12 - self.xi2m2_10_loop12
        self.Y10loop12 = 3*self.xi2m2_10_loop12
    
        # the various Us
        self.U3 = self.xi_l_n(1,-1, _int=-5./21*self.R1)
        self.U11 = self.xi_l_n(1,-1,-6./7*(self.R1+self.R2))
        self.U20 = self.xi_l_n(1,-1,-3./7*self.Q8)
        self.Us2 = self.xi_l_n(1,-1,-1./7*self.Qs2) # earlier this was 2/7 but that's wrong
    
    def setup_third_order(self):
        # All the terms involving the third order bias, which is really just a few
        
        P3_0 = self.k**2 * self.template_QR(0, 24./5*self.xi00/self.qv)
        P3_1 = self.k    * self.template_QR(1, -16./5*self.xi11/self.qv)
        P3_2 = self.k**2 * self.template_QR(2, -20./7*self.xi20/self.qv) + self.template_QR(2,4.*self.xi22/self.qv)
        P3_3 = self.k    * self.template_QR(3, -24./5*self.xi31/self.qv)
        P3_4 = self.k**2 * self.template_QR(4, 72./35*self.xi40/self.qv)

        self.Rb3 = 2 * 2./63 * (P3_0 + P3_1 + P3_2 + P3_3 + P3_4)
        
        self.theta = self.xi_l_n(0,0, _int= self.p * self.Rb3)
        self.Ub3 = - self.xi_l_n(1,-1, _int= self.p * self.Rb3)
        
    
    def xi_l_n(self, l, n, _int=None, IR_cut = 'all', extrap=False, qmin=1e-3, qmax=1000, side='both'):
        '''
        Calculates the generalized correlation function xi_l_n, which is xi when l = n = 0
        
        If _int is None assume integrating the power spectrum.
        '''
        if _int is None:
            integrand = self.p * self.k**n
        else:
            integrand = _int * self.k**n
        
        if IR_cut is not 'all':
            if IR_cut == 'gt':
                integrand *= self.ir_greater
            elif IR_cut == 'lt':
                integrand *= self.ir_less
        
        qs, xint =  self.sph.sph(l,integrand)

        if extrap:
            qrange = (qs > qmin) * (qs < qmax)
            return loginterp(qs[qrange],xint[qrange],side=side)(self.qv)
        else:
            return np.interp(self.qv, qs, xint)

    def template_QR(self,l,integrand):
        '''
        Interpolates the Hankel transformed R(k), Q(k) back onto self.k
        '''
        kQR, QR = self.sphr.sph(l,integrand)
        return np.interp(self.k, kQR, QR)

