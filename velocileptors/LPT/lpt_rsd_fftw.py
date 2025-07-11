import numpy as np
import os

from scipy.special import hyp2f1, gamma
from scipy.interpolate import interp1d

from velocileptors.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SphericalBesselTransformNP
from velocileptors.Utils.loginterp import loginterp
from velocileptors.Utils.gaussian_poly_extrap import gaussian_poly_extrap

from velocileptors.Utils.qfuncfft import QFuncFFT



class LPT_RSD:

    '''
        Class to evaluate the one-loop power spectrum in redshift space with the
        linear velocities resummed. See arXiv:XXXX
        
        Throughout this code we refer to mu_q as "mu" and mu = n.k as "nu."
    '''

    def __init__(self, k, p,\
                use_Pzel = True, third_order = True, shear=True, one_loop=True,\
                kIR = None, cutoff=10, jn=5, N = 2000, threads=None, extrap_min = -5, extrap_max = 3):

        self.N = N
        self.extrap_max = extrap_max
        self.extrap_min = extrap_min
        
        self.kIR = kIR
        self.cutoff = cutoff
        self.kint = np.logspace(extrap_min,extrap_max,self.N)
        self.qint = np.logspace(-extrap_max,-extrap_min,self.N)
        
        self.third_order = third_order
        self.shear = shear or third_order
        self.one_loop = one_loop
        
        self.k = k
        self.p = p
        self.pint = loginterp(k,p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.setup_powerspectrum()
        
        self.pktables = {}
        
        if self.third_order:
            self.num_power_components = 13
        elif self.shear:
            self.num_power_components = 11
        else:
            self.num_power_components = 7
        
        self.jn = jn
        
        if threads is None:
            self.threads = int( os.getenv("OMP_NUM_THREADS","1") )
        else:
            self.threads = threads

        self.sph = SphericalBesselTransform(self.qint, L=self.jn, ncol=self.num_power_components, threads=self.threads)
        self.sph1 = SphericalBesselTransform(self.qint, L=self.jn, ncol=1, threads=self.threads)
        self.sphr = SphericalBesselTransformNP(self.kint,L=5,fourier=True)
        
        # Use Pzel vs Pb1^2 for the counterterms
        self.use_Pzel = use_Pzel

    
    def setup_powerspectrum(self):
    
        # This sets up terms up to one loop in the combination (symmetry factors) they appear in pk
    
        self.qf = QFuncFFT(self.kint, self.pint, kIR=self.kIR, qv=self.qint, oneloop=self.one_loop, shear=self.shear, third_order=self.third_order)
    
        # linear terms
        self.Xlin = self.qf.Xlin_lt
        self.Ylin = self.qf.Ylin_lt
        self.XYlin = self.Xlin + self.Ylin; self.sigma = self.XYlin[-1]
        self.yq = self.Ylin / self.qint
        
        self.Xlin_gt = self.qf.Xlin_gt
        self.Ylin_gt = self.qf.Ylin_gt
        
        self.Ulin = self.qf.Ulin
        self.corlin = self.qf.corlin
        
        # load shear functions
        if self.shear:
            self.Xs2 = self.qf.Xs2
            self.Ys2 = self.qf.Ys2; self.sigmas2 = (self.Xs2 + self.Ys2)[-1]
            self.V = self.qf.V
            self.zeta = self.qf.zeta
            self.chi = self.qf.chi
            
        if self.one_loop:

            self.X13, self.Y13 = self.qf.Xloop13, self.qf.Yloop13
            self.X22, self.Y22 = self.qf.Xloop22, self.qf.Yloop22
            
            # These are the decomposition for W112, which we need independently
            self.V1, self.V3 = self.qf.V1loop112, self.qf.V3loop112
            self.T = self.qf.Tloop112
        
            self.X10 = 2 * self.qf.X10loop12
            self.Y10 = 2 * self.qf.Y10loop12
            self.sigma10 = (self.X10 + self.Y10)[-1]
        
            self.U3 = self.qf.U3
            self.U11 = self.qf.U11
            self.U20 = self.qf.U20
            self.Us2 = self.qf.Us2
            self.Ub3 = self.qf.Ub3
            self.theta = self.qf.theta

        else:
            self.X13, self.Y13, self.X22, self.Y22, self.sigmaloop, self.V1, self.V3, self.T, self.Tloop, self.X10, self.Y10, self.sigma10, self.U3, self.U11, self.U20, self.Us2, self.Ub3, self.theta = (0,)*18
            
    #### Define RSD Kernels #######
    
    def setup_rsd_facs(self,f,nu,D=1,nmax=10):
    
        self.f = f
        self.nu = nu
        self.D = D
        self.Kfac = np.sqrt(1+f*(2+f)*nu**2); self.Kfac2 = self.Kfac**2
        self.s = f*nu*np.sqrt(1-nu**2)/self.Kfac
        self.c = np.sqrt(1-self.s**2); self.c2 = self.c**2; self.ic2 = 1/self.c2; self.c3 = self.c**3
        self.Bfac = -0.5 * self.Kfac2 * self.Ylin * self.D**2 # this times k is "B"
        
        # Define Anu, Bnu such that \hn \cdot \hq = Anu * mu + Bnu * sqrt(1-mu^2) cos(phi)
        self.Anu, self.Bnu = self.nu * (1 + f) / self.Kfac, np.sqrt(1-self.nu**2) / self.Kfac
        
        # Compute derivatives
        # Each is a function of f, nu times (kq)^(-n) for the nth derivative
        
        # and the hypergeometric functions
        self.hyp1 = np.zeros( (self.jn+nmax, self.jn+nmax))
        self.hyp2 = np.zeros( (self.jn+nmax, self.jn+nmax))
        self.fnms = np.zeros( (self.jn+nmax, self.jn+nmax))
        
        for n in range(self.jn+nmax):
            for m in range(self.jn+nmax):
                self.hyp1[n,m] = hyp2f1(0.5-n,-n,0.5-m-n,self.ic2)
                self.hyp2[n,m] = hyp2f1(1.5-n,-n,0.5-m-n,self.ic2)
                self.fnms[n,m] = gamma(m+n+0.5)/gamma(m+1)/gamma(n+0.5)/gamma(1-m+n)
        
        self.G0_l_ns = np.zeros( (self.jn,nmax) )
        self.dG0dA_l_ns = np.zeros( (self.jn,nmax) )
        self.d2G0dA2_l_ns = np.zeros( (self.jn,nmax) )
        self.dG0dC_l_ns = np.zeros( (self.jn,nmax) )
        self.d2G0dCdA_l_ns = np.zeros( (self.jn,nmax) )
        self.d2G0dC2_l_ns = np.zeros( (self.jn,nmax) )
        self.d3G0dA3_l_ns = np.zeros( (self.jn,nmax) )
        self.d3G0dCdA2_l_ns = np.zeros( (self.jn,nmax) )
        self.d4G0dA4_l_ns = np.zeros( (self.jn,nmax) )
        
        for ll in range(self.jn):
            for nn in range(nmax):
                self.G0_l_ns[ll,nn] = self._G0_l_n(ll+nn,ll)
                self.dG0dA_l_ns[ll,nn] = self._dG0dA_l_n(ll+nn,ll)
                self.d2G0dA2_l_ns[ll,nn] = self._d2G0dA2_l_n(ll+nn,ll)
                
                # One loop terms
                self.dG0dC_l_ns[ll,nn] = self._dG0dC_l_n(ll+nn,ll)
                self.d2G0dCdA_l_ns[ll,nn] = self._d2G0dCdA_l_n(ll+nn,ll)
                self.d2G0dC2_l_ns[ll,nn] = self._d2G0dC2_l_n(ll+nn,ll)
                self.d3G0dA3_l_ns[ll,nn] = self._d3G0dA3_l_n(ll+nn,ll)
                self.d3G0dCdA2_l_ns[ll,nn] = self._d3G0dCdA2_l_n(ll+nn,ll)
                self.d4G0dA4_l_ns[ll,nn] = self._d4G0dA4_l_n(ll+nn,ll)
                
        # Also precompute the (BA^2/rho^2) factor
        self.powerfacs = np.array([ (self.Bfac /self.ic2)**n for n in range(self.jn + nmax) ]) # does not include factor of k^2n
        

        
    
    def _G0_l_n(self,n,m):
        x = self.ic2

        return  self.fnms[n,m] * self.hyp1[n,m]
    
    
    def _dG0dA_l_n(self,n,m):
        # Note that in the derivatives we omit factors of (kq)^n left in comments for speedier vector evaluation later
    
        x = self.ic2

        ret = self.s * (-self.hyp1[n,m] + (1-2*n)*self.hyp2[n,m])
        ret *= - self.s
        
        return self.fnms[n,m] * ret # / (k*self.qint)
    
    def _d2G0dA2_l_n(self,n,m):
        x = self.ic2
        
        ret = (1-1./x) * ( (2*m-1-4*n*(m+1))*self.hyp1[n,m] \
                                                +(1-4*n**2+m*(4*n-2))*self.hyp2[n,m] )
        return self.fnms[n,m] * ret #/(k*self.qint)**2
        
    def _dG0dC_l_n(self,n,m):
        x = self.ic2

        ret = self.s * (-self.hyp1[n,m] + (1-2*n)*self.hyp2[n,m])
        
        return self.fnms[n,m] * ret # / (k*self.qint)
        
    def _d2G0dCdA_l_n(self,n,m):
        x = self.ic2
        
        ret  = - ( 2*(m - 2*n*(1+m))*self.c**2 + self.s**2 ) * self.hyp1[n,m]
        ret += (1-2*n) * ( 2*(m-n)*self.c**2 + self.s**2 ) * self.hyp2[n,m]
        
        ret *= self.s / self.c
        
        return self.fnms[n,m] * ret # /(k*self.qint)**2
        
    def _d2G0dC2_l_n(self,n,m):
        x = self.ic2

        ret  = ( (1+2*m-4*n*(1+m))*self.c**2 + 2*self.s**2 ) * self.hyp1[n,m]
        ret += -(1-2*n) * ( (1+2*m-2*n)*self.c**2 + 2*self.s**2 ) * self.hyp2[n,m]
                
        return self.fnms[n,m] * ret # / (k*self.qint)**2
        
    def _d3G0dA3_l_n(self,n,m):
        x = self.ic2
        
        coeff1A = 2*(1-m)*(1-2*m) + 8*(2-m)*(1+m)*n + 8*n**2*(1+m)
        coeff1C = - (1-2*m+4*n*(1+m))
        ret = (coeff1A * self.c**2 + coeff1C * self.s**2) * self.hyp1[n,m]
        
        coeff2A = -(1-2*n)*( 2*(1-2*m+2*n)*(1-m+n) )
        coeff2C = (1-2*n)*(1-2*m+4*n*(1+m))
        ret += (coeff2A * self.c**2 + coeff2C * self.s**2) * self.hyp2[n,m]

        ret *= (self.s**2/self.c)
        
        return self.fnms[n,m] * ret # / (k*self.qint)**3
        
    
    def _d4G0dA4_l_n(self,n,m):
        x = self.ic2
        
        coeff1A = -6 + 22*m - 24*m**2 + 8*m**3 \
                 + n*(-76 - 28*m + 32*m**2 - 16*m**3) \
                 + n**2 * (-56 - 24*m + 32*m**2 ) + n**3 * ( -16 - 16*m )
        coeff1C = 9 - 24*m + 12*m**2 + n * (56 + 24*m - 32*m**2) +\
                   n**2 * (32 + 48*m + 16*m**2)
        ret = (coeff1A * self.c**2 + coeff1C * self.s**2) * self.hyp1[n,m]
        
        coeff2A = 2*(-3+2*m-2*n)*(1-2*m+2*n)*(1-m+n)
        coeff2C = 9 - 24*m + 12*m**2 + n*(44 + 8*m - 16*m**2) + n**2 * (20 + 16*m)
        ret += -(1-2*n)*(coeff2A * self.c**2 + coeff2C * self.s**2) * self.hyp2[n,m]

        ret *= self.fnms[n,m] * self.s**2 # / (k*self.qint)**4
        
        return ret

        
    # dG/dA^2dC
    def _d3G0dCdA2_l_n(self,n,m):
        x = self.ic2
        
        coeff1 =  2 * (m-2*m**2-4*n*(1-m**2)-4*n**2*(1+m)) * self.c**2
        coeff1 += 3 * (1-2*m+4*n*(1+m)) * self.s**2
        
        coeff2 = 2 * (1-2*m+2*n)*(m-n) * self.c**2
        coeff2 += (3-6*m+8*n+4*m*n) * self.s**2
        coeff2 *= - (1-2*n)
        
        ret  = coeff1 * self.hyp1[n,m]
        ret += coeff2 * self.hyp2[n,m]
        
        ret *= self.s
        
        return self.fnms[n,m] * ret # / (k*self.qint)**3
    

    def _G0_l(self,l,k, nmax=10):
        
        summand =  (k**(2* (l+np.arange(nmax))) * self.G0_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0)
        
        
    
    def _dG0dA_l(self,l,k,nmax=10):
        
        summand =  (k**(2* (l+np.arange(nmax))) * self.dG0dA_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)
    
    
    def _d2G0dA2_l(self,l,k,nmax=10):

        
        summand =  (k**(2* (l+np.arange(nmax))) * self.d2G0dA2_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**2
        
    def _dG0dC_l(self,l,k,nmax=10):

        
        summand =  (k**(2* (l+np.arange(nmax))) * self.dG0dC_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)
        
        
    def _d2G0dCdA_l(self,l,k,nmax=10):

        summand =  (k**(2* (l+np.arange(nmax))) * self.d2G0dCdA_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**2
        
    def _d2G0dC2_l(self,l,k,nmax=10):
    
        summand =  (k**(2* (l+np.arange(nmax))) * self.d2G0dC2_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**2
        
    def _d3G0dA3_l(self,l,k,nmax=10):

        summand =  (k**(2* (l+np.arange(nmax))) * self.d3G0dA3_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**3
    
    def _d3G0dCdA2_l(self,l,k,nmax=10):

        summand =  (k**(2* (l+np.arange(nmax))) * self.d3G0dCdA2_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**3
        
        
    def _d4G0dA4_l(self,l,k,nmax=10):

        summand =  (k**(2* (l+np.arange(nmax))) * self.d4G0dA4_l_ns[l,:nmax])[:,None] * self.powerfacs[l:l+nmax,:]
        return np.sum(summand,axis=0) / (k*self.qint)**4
        
    
    ### Now define the actual integrals!

    def p_integrals(self, k, nmax=8, old=False):
        
        ksq = k**2
        Kfac = self.Kfac
        f = self.f
        nu = self.nu
        Anu, Bnu = self.Anu, self.Bnu
        
        K = k*self.Kfac; Ksq = K**2
        Knfac = nu*(1+f)
        
        D2 = self.D**2; D4 = D2**2

        expon = np.exp(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma)
            
            
        A = k*self.qint*self.c
        C = k*self.qint*self.s
        
        
        G0s =  [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0] + [0] + [0]
        dGdAs =  [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        dGdCs = [self._dG0dC_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        d2GdA2s = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0]
        d2GdCdAs = [self._d2G0dCdA_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d2GdC2s = [self._d2G0dC2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d3GdA3s = [self._d3G0dA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdCdA2s = [self._d3G0dCdA2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d4GdA4s = [self._d4G0dA4_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
                
        G01s = [-(dGdAs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G02s = [-(d2GdA2s[ii] + A * dGdAs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        G03s = [d3GdA3s[ii] + 1.5*A*d2GdA2s[ii-1] + 1.5*dGdAs[ii-1] \
                 + 0.75*A**2*dGdAs[ii-2] + 0.75*A*G0s[ii-2] + A**3/8.*G0s[ii-3] for ii in range(self.jn)]
        G04s = [d4GdA4s[ii] + 2*A*d3GdA3s[ii-1] + 3*d2GdA2s[ii-1] \
                + 1.5*A**2*d2GdA2s[ii-2] + 3*A*dGdAs[ii-2] + 0.75*G0s[ii-2]\
                + 0.5*A**3*dGdAs[ii-3] + 0.75*A**2*G0s[ii-3]\
                + A**4/16. * G0s[ii-4] for ii in range(self.jn)]
                 
        G10s = [ dGdCs[ii] + 0.5*C*G0s[ii-1]  for ii in range(self.jn)]
        
        G11s = [ d2GdCdAs[ii] + 0.5*C*dGdAs[ii-1] + 0.5*A*dGdCs[ii-1] + 0.25*A*C*G0s[ii-2] for ii in range(self.jn)]
        G20s = [-(d2GdC2s[ii] + C * dGdCs[ii-1] + 0.5*G0s[ii-1] + 0.25 * C**2 *G0s[ii-2]) for ii in range(self.jn)]
        G12s = [-(d3GdCdA2s[ii] + 0.5*C*d2GdA2s[ii-1] + A*d2GdCdAs[ii-1] + 0.5*dGdCs[ii-1]\
                  + 0.5*A*C*dGdAs[ii-2] + 0.25*A**2*dGdCs[ii-2] + 0.25*C*G0s[ii-2] + A**2*C/8*G0s[ii-3])  for ii in range(self.jn)]

        ret = np.zeros(self.num_power_components)
            
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
                            
        for l in range(self.jn):
            
            mu0 = G0s[l]
            nq1 = self.Anu * G01s[l] + self.Bnu * G10s[l]
            mu_nq1 = self.Anu * G02s[l] + self.Bnu * G11s[l]
            nq2 = self.Anu**2 * G02s[l] + 2 * self.Anu * self.Bnu * G11s[l] + self.Bnu**(2+2*old) * G20s[l]
            mu1 = G01s[l]
            mu2 = G02s[l]
            mu3 = G03s[l]
            mu2_nq1 = self.Anu * G03s[l] + self.Bnu * G12s[l]
            mu4 = G04s[l]
            
            bias_integrands[0,:] = 1 * G0s[l]  - 0.5 * Ksq * (self.Xlin_gt * G0s[l] + self.Ylin_gt * mu2) # za
            
            bias_integrands[0,:] += -0.5 * ksq * ( 2*(Kfac**2 + 2*f*(1+f)*nu**2) * G0s[l] * self.X13 +\
                                                  2*(Kfac**2*mu2 + 2*f*Kfac*nu*mu_nq1) * self.Y13 +\
                                                   (Kfac**2 + 2*f*(1+f)*nu**2 + f**2*nu**2) * G0s[l] * self.X22 +\
                                                   (Kfac**2*mu2 + 2*f*Kfac*nu*mu_nq1 + f**2*nu**2*nq2) * self.Y22) \
                                  + Ksq**2 / 8. * (self.Xlin_gt**2 * G0s[l] + 2*self.Xlin_gt*self.Ylin_gt*mu2 + self.Ylin_gt**2 * mu4)# Aloop

                                            
            bias_integrands[0,:] += 0.5*k**3 * ( 2*Kfac*(Kfac**2+f*(1+f)*nu**2) * G01s[l] * self.V1 +  \
                                                Kfac**2 * (Kfac*G01s[l] + f*nu*nq1) * self.V3 + \
                                                Kfac**2 * (Kfac*G03s[l] + f*nu*mu2_nq1) * self.T)
                                                
            bias_integrands[1,:] = -2 * K * (self.Ulin + self.U3) * mu1 - Ksq * (self.X10 * mu0 + self.Y10 * mu2 ) \
                                   -4*f*k*nu*self.U3*nq1 - f*ksq*nu*(self.X10 * Knfac * mu0 + Kfac * self.Y10 * mu_nq1)\
                                   -2 * K * self.Ulin * ( -0.5*Ksq*(self.Xlin_gt*mu1 + self.Ylin_gt*mu3) )
                        
            bias_integrands[2,:] = self.corlin * (mu0 - 0.5*Ksq*(self.Xlin_gt*mu0 + self.Ylin_gt*mu2) )\
                                   - Ksq*self.Ulin**2*mu2 - (K*mu1 + f*k*nu*nq1)*self.U11
                                   
                                   
            bias_integrands[3,:] = - Ksq * self.Ulin**2 * mu2 - k*(Kfac*mu1 + f*nu*nq1)*self.U20 # b2
            bias_integrands[4,:] = -2 * K * self.Ulin * self.corlin * mu1 # b1b2
            bias_integrands[5,:] = 0.5 * self.corlin**2 * mu0 # b2sq
            
            if self.shear or self.third_order:
                bias_integrands[6,:] = - Ksq * (self.Xs2 * mu0 + self.Ys2 * mu2) - 2*k*(Kfac*mu1 + f*nu*nq1)*self.Us2 # bs should be both minus
                bias_integrands[7,:] = -2*K*self.V * mu1 # b1bs
                bias_integrands[8,:] = self.chi * mu0 # b2bs
                bias_integrands[9,:] = self.zeta * mu0 # bssq
                
            if self.third_order:
                bias_integrands[10,:] = -2 * K * self.Ub3 * mu1 #b3
                bias_integrands[11,:] = 2 * self.theta * mu0 #b1 b3
            
            if self.use_Pzel:
                bias_integrands[-1,:] = 1 * G0s[l] - 0.5 * Ksq * (self.Xlin_gt * G0s[l] + self.Ylin_gt * mu2) # za
            else:
                bias_integrands[-1,:] = self.corlin * mu0
                                   
            # multiply by IR exponent
            if l == 0:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                bias_integrands -= bias_integrands[:,-1][:,None]
            else:
                bias_integrands = bias_integrands * expon * (-2./k/self.qint)**l
                                                                
            # do FFTLog
            ktemps, bias_ffts = self.sph.sph(l, bias_integrands)
            ret += interp1d(ktemps, bias_ffts)(k)

        return 4*suppress*np.pi*ret
        

    def make_ptable(self, f, nu, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50,nmax=5, old=False):
    
        self.setup_rsd_facs(f,nu,nmax=nmax)
        
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
            
        self.pktable = np.zeros([nk, self.num_power_components+1]) # one column for ks
        
        self.pktable[:, 0] = kv[:]
        for foo in range(nk):
            self.pktable[foo, 1:] = self.p_integrals(kv[foo],nmax=nmax, old=old)
        
        # store a copy in pktables dictionary
        self.pktables[nu] = np.array(self.pktable)
        



    def make_pltable(self,f, apar = 1, aperp = 1, ngauss = 3, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50, nmax=8):
        '''
        Make a table of the monopole and quadrupole in k space.
        Uses gauss legendre integration.
            
        '''
        
        # since we are always symmetric in nu, can ignore negative values
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        #self.pknutable = np.zeros((len(nus),nk,self.num_power_components+3)) # counterterms have distinct nu structure
        # counterterms + stoch terms have distinct nu structure and have to be added here
        # e.g. k^2 mu^2 is not the same as k_obs^2 mu_obs^2!
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
        self.pknutable = np.zeros((len(nus),nk,self.num_power_components+6)) 
        
        
        # To implement AP:
        # Calculate P(k,nu) at the true coordinates, given by
        # k_true = k_apfac * kobs
        # nu_true = nu * a_perp/a_par/fac
        # Note that the integration grid on the other hand is never observed
        
        for ii, nu in enumerate(nus_calc):
        
            fac = np.sqrt(1 + nu**2 * ((aperp/apar)**2-1))
            k_apfac = fac / aperp
            nu_true = nu * aperp/apar/fac
            vol_fac = apar * aperp**2
        
            self.setup_rsd_facs(f,nu_true,nmax=nmax)
            
            for jj, k in enumerate(kv):
                ktrue = k_apfac * k
                pterms = self.p_integrals(ktrue,nmax=nmax)
                
                #self.pknutable[ii,jj,:-4] = pterms[:-1]
                self.pknutable[ii,jj,:-7] = pterms[:-1]
                
                # counterterms
                
                #self.pknutable[ii,jj,-4] = ktrue**2 * pterms[-1]
                #self.pknutable[ii,jj,-3] = ktrue**2 * nu_true**2 * pterms[-1]
                #self.pknutable[ii,jj,-2] = ktrue**2 * nu_true**4 * pterms[-1]
                #self.pknutable[ii,jj,-1] = ktrue**2 * nu_true**6 * pterms[-1]
                
                self.pknutable[ii,jj,-7] = ktrue**2 * pterms[-1]
                self.pknutable[ii,jj,-6] = ktrue**2 * nu_true**2 * pterms[-1]
                self.pknutable[ii,jj,-5] = ktrue**2 * nu_true**4 * pterms[-1]
                self.pknutable[ii,jj,-4] = ktrue**2 * nu_true**6 * pterms[-1]
                
                # stochastic terms
                self.pknutable[ii,jj,-3] = 1
                self.pknutable[ii,jj,-2] = ktrue**2 * nu_true**2
                self.pknutable[ii,jj,-1] = ktrue**4 * nu_true**4
        
        self.pknutable[ngauss:,:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.kv = kv
        self.p0ktable = 0.5 * np.sum((ws*L0)[:,None,None]*self.pknutable,axis=0) / vol_fac
        self.p2ktable = 2.5 * np.sum((ws*L2)[:,None,None]*self.pknutable,axis=0) / vol_fac
        self.p4ktable = 4.5 * np.sum((ws*L4)[:,None,None]*self.pknutable,axis=0) / vol_fac
        
        return 0

    def combine_bias_terms_pkmu(self,nu,bvec):
        '''
        Combine bias terms into P(k,nu) given the bias paramters and counterterms listed below.
        
        Returns k, pknu.
        '''

        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6, sn,sn2,sn4 = bvec
        bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        
        try:
            pknu = self.pktables[nu]
        except:
            print("ERROR: Use make_ptable first to compute power spectrum components at angle nu.")
            return np.nan, np.nan
    
        kv = pknu[:,0]; za = pknu[:,-1]
        pktemp = np.copy(pknu)[:,1:-1]
                    
        res = np.sum(pktemp * bias_monomials,axis=1)\
              + (alpha0 + alpha2*nu**2 + alpha4*nu**4 + alpha6*nu**6) * kv**2 * za\
            + sn + sn2 * kv**2*nu**2 + sn4 * kv**4 * nu**4
                    
        return kv, res
        
        
    def combine_bias_terms_pkell(self,bvec):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec
        #bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3, alpha0, alpha2, alpha4,alpha6])
        bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3, alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        try:
            kv = self.kv
            p0 = np.sum(self.p0ktable * bias_monomials,axis=1)# + sn + 1./3 * kv**2 * sn2 + 1./5 * kv**4 * sn4
            p2 = np.sum(self.p2ktable * bias_monomials,axis=1)# + 2 * kv**2 * sn2 / 3 + 4./7 * kv**4 * sn4
            p4 = np.sum(self.p4ktable * bias_monomials,axis=1)# + 8./35 * kv**4 * sn4
            return kv, p0, p2, p4
        except:
            print("First generate multipole table with make_pltable.")
            
            
    def combine_bias_terms_xiell(self,bvec,method='loginterp'):
        '''
        Same as above but further transform the pkells into xiells.
        
        Again, the paramters f, AP are assumed to be what was input into p{ell}ktable.
        
        '''
        
        kv, p0, p2, p4 = self.combine_bias_terms_pkell(bvec)
        
        if method == 'loginterp':
        
            damping = np.exp(-(self.kint/10)**2)
            p0int = loginterp(kv, p0)(self.kint) * damping
            p2int = loginterp(kv, p2)(self.kint) * damping
            p4int = loginterp(kv, p4)(self.kint) * damping
            
        elif method == 'gauss_poly':
            # Add a point at k = 0 to the spline in k taper nicely
            
            frac = 1
            
            p0int = gaussian_poly_extrap( self.kint,\
                                          np.concatenate(([0], kv)),\
                                          np.concatenate(([0], p0)), frac=frac)
            
            p2int = gaussian_poly_extrap( self.kint,\
                                          np.concatenate(([0], kv)),\
                                          np.concatenate(([0], p2)), frac=frac )
            
            p4int = gaussian_poly_extrap( self.kint,\
                                          np.concatenate(([0], kv)),\
                                          np.concatenate(([0], p4)), frac=frac )
            
        elif method == 'min_cut':
            # Start log extrapolating when p_ell is below a threshold value:
            ftol = 1e-4
            damping = np.exp(-(self.kint/10)**2)
            
            pints = [np.zeros_like(self.kint), np.zeros_like(self.kint), np.zeros_like(self.kint),]
            
            for ii, pp in enumerate([p0,p2,p4]):
                
                iis = np.arange(len(kv))
                pval = np.max(pp)
                
                try:
                    zero_crossing = np.where(np.diff(np.sign(pp)))[0][0]
                except:
                    zero_crossing = len(pp)
                    
                cross_min = pp > (ftol * pval)

                # union is where we interpolate
                where_int = (iis < zero_crossing) * cross_min
                ktemp, ptemp = kv[where_int], pp[where_int]

                pints[ii] += loginterp(ktemp, ptemp)(self.kint) * damping

            p0int, p2int, p4int = pints
            
            
        ss0, xi0 = self.sphr.sph(0,p0int)
        ss2, xi2 = self.sphr.sph(2,p2int); xi2 *= -1
        ss4, xi4 = self.sphr.sph(4,p4int)
        
        return (ss0, xi0), (ss2, xi2), (ss4, xi4)
        
        #except:
        #    print("First generate multipole table with make_pltable.")
            
            
            

    ### Alternative functions to first combine bias terms, then compute power spectrum
    ### This set of functions currently assumes nonzero bs and b3
    
    
    def p_integral_fixedbias(self, k, bvec, nmax=8):
        
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec
        bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3])
        
        ksq = k**2
        Kfac = self.Kfac
        f = self.f
        nu = self.nu
        Anu, Bnu = self.Anu, self.Bnu
        
        K = k*self.Kfac; Ksq = K**2
        Knfac = nu*(1+f)
        
        D2 = self.D**2; D4 = D2**2

        expon = np.exp(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        exponm1 = np.expm1(-0.5*Ksq * D2* (self.XYlin - self.sigma))
        suppress = np.exp(-0.5*Ksq * D2* self.sigma)
            
            
        A = k*self.qint*self.c
        C = k*self.qint*self.s
        
        
        G0s =  [self._G0_l(ii,k,nmax=nmax)    for ii in range(self.jn)] + [0] + [0] + [0] + [0]
        dGdAs =  [self._dG0dA_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        dGdCs = [self._dG0dC_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0] + [0]
        d2GdA2s = [self._d2G0dA2_l(ii,k,nmax=nmax) for ii in range(self.jn)] + [0] + [0]
        d2GdCdAs = [self._d2G0dCdA_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d2GdC2s = [self._d2G0dC2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0] + [0]
        d3GdA3s = [self._d3G0dA3_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d3GdCdA2s = [self._d3G0dCdA2_l(ii,k,nmax=nmax) for ii in range(self.jn) ] + [0]
        d4GdA4s = [self._d4G0dA4_l(ii,k,nmax=nmax) for ii in range(self.jn) ]
                
        G01s = [-(dGdAs[ii] + 0.5*A*G0s[ii-1])   for ii in range(self.jn)]
        G02s = [-(d2GdA2s[ii] + A * dGdAs[ii-1] + 0.5*G0s[ii-1] + 0.25 * A**2 *G0s[ii-2]) for ii in range(self.jn)]
        G03s = [d3GdA3s[ii] + 1.5*A*d2GdA2s[ii-1] + 1.5*dGdAs[ii-1] \
                 + 0.75*A**2*dGdAs[ii-2] + 0.75*A*G0s[ii-2] + A**3/8.*G0s[ii-3] for ii in range(self.jn)]
        G04s = [d4GdA4s[ii] + 2*A*d3GdA3s[ii-1] + 3*d2GdA2s[ii-1] \
                + 1.5*A**2*d2GdA2s[ii-2] + 3*A*dGdAs[ii-2] + 0.75*G0s[ii-2]\
                + 0.5*A**3*dGdAs[ii-3] + 0.75*A**2*G0s[ii-3]\
                + A**4/16. * G0s[ii-4] for ii in range(self.jn)]
                 
        G10s = [ dGdCs[ii] + 0.5*C*G0s[ii-1]  for ii in range(self.jn)]
        
        G11s = [ d2GdCdAs[ii] + 0.5*C*dGdAs[ii-1] + 0.5*A*dGdCs[ii-1] + 0.25*A*C*G0s[ii-2] for ii in range(self.jn)]
        G20s = [-(d2GdC2s[ii] + C * dGdCs[ii-1] + 0.5*G0s[ii-1] + 0.25 * C**2 *G0s[ii-2]) for ii in range(self.jn)]
        G12s = [-(d3GdCdA2s[ii] + 0.5*C*d2GdA2s[ii-1] + A*d2GdCdAs[ii-1] + 0.5*dGdCs[ii-1]\
                  + 0.5*A*C*dGdAs[ii-2] + 0.25*A**2*dGdCs[ii-2] + 0.25*C*G0s[ii-2] + A**2*C/8*G0s[ii-3])  for ii in range(self.jn)]

        ret = 0
        bias_integrands = np.zeros( (self.num_power_components,self.N)  )
        bias_integrand  = np.zeros(self.N)
                            
        for l in range(self.jn):
            
            mu0 = G0s[l]
            nq1 = self.Anu * G01s[l] + self.Bnu * G10s[l]
            mu_nq1 = self.Anu * G02s[l] + self.Bnu * G11s[l]
            nq2 = self.Anu**2 * G02s[l] + 2 * self.Anu * self.Bnu * G11s[l] + self.Bnu**2 * G20s[l]
            mu1 = G01s[l]
            mu2 = G02s[l]
            mu3 = G03s[l]
            mu2_nq1 = self.Anu * G03s[l] + self.Bnu * G12s[l]
            mu4 = G04s[l]
            
            bias_integrands[0,:] = 1 * G0s[l] - 0.5 * Ksq * (self.Xlin_gt * G0s[l] + self.Ylin_gt * mu2) # za
            
            bias_integrands[0,:] += -0.5 * ksq * ( 2*(Kfac**2 + 2*f*(1+f)*nu**2) * G0s[l] * self.X13 +\
                                                  2*(Kfac**2*mu2 + 2*f*Kfac*nu*mu_nq1) * self.Y13 +\
                                                   (Kfac**2 + 2*f*(1+f)*nu**2 + f**2*nu**2) * G0s[l] * self.X22 +\
                                                   (Kfac**2*mu2 + 2*f*Kfac*nu*mu_nq1 + f**2*nu**2*nq2) * self.Y22)\
                                 + Ksq**2 / 8. * (self.Xlin_gt**2 * G0s[l] + 2*self.Xlin_gt*self.Ylin_gt*mu2 + self.Ylin_gt**2 * mu4)# Aloop

                                            
            bias_integrands[0,:] += 0.5*k**3 * ( 2*Kfac*(Kfac**2+f*(1+f)*nu**2) * G01s[l] * self.V1 +  \
                                                Kfac**2 * (Kfac*G01s[l] + f*nu*nq1) * self.V3 + \
                                                Kfac**2 * (Kfac*G03s[l] + f*nu*mu2_nq1) * self.T)
                                                
            bias_integrands[1,:] = -2 * K * (self.Ulin + self.U3) * mu1 - Ksq * (self.X10 * mu0 + self.Y10 * mu2 ) \
                                   -4*f*k*nu*self.U3*nq1 - f*ksq*nu*(self.X10 * Knfac * mu0 + Kfac * self.Y10 * mu_nq1)\
                                   -2 * K * self.Ulin * ( -0.5*Ksq*(self.Xlin_gt*mu1 + self.Ylin_gt*mu3) )
                                   
            bias_integrands[2,:] = self.corlin * (mu0 - 0.5*Ksq*(self.Xlin_gt*mu0 + self.Ylin_gt*mu2) )\
                                   - Ksq*self.Ulin**2*mu2 - k*(Kfac*mu1 + f*k*nu*nq1)*self.U11
                                   
                                   
            bias_integrands[3,:] = - Ksq * self.Ulin**2 * mu2 - k*(Kfac*mu1 + f*nu*nq1)*self.U20 # b2
            bias_integrands[4,:] = -2 * K * self.Ulin * self.corlin * mu1 # b1b2
            bias_integrands[5,:] = 0.5 * self.corlin**2 * mu0 # b2sq
            
            if self.shear or self.third_order:
                bias_integrands[6,:] = - Ksq * (self.Xs2 * mu0 + self.Ys2 * mu2) - 2*k*(Kfac*mu1 + f*nu*nq1)*self.Us2 # bs should be both minus
                bias_integrands[7,:] = -2*K*self.V * mu1 # b1bs
                bias_integrands[8,:] = self.chi * mu0 # b2bs
                bias_integrands[9,:] = self.zeta * mu0 # bssq
                
            if self.third_order:
                bias_integrands[10,:] = -2 * K * self.Ub3 * mu1 #b3
                bias_integrands[11,:] = 2 * self.theta * mu0 #b1 b3
            
            if self.use_Pzel:
                bias_integrands[-1,:] = 1 * G0s[l] - 0.5 * Ksq * (self.Xlin_gt * G0s[l] + self.Ylin_gt * mu2) # za
            else:
                bias_integrands[-1,:] = self.corlin * mu0
                
            # sum up bias terms, treating counterterms separately
            bias_integrand  = np.sum( bias_monomials[:,None]*bias_integrands[:-1,:],axis=0 )
            bias_integrand += k**2 * (alpha0 + alpha2*nu**2 + alpha4*nu**4 + alpha6*nu**6) * bias_integrands[-1,:]
            
            # multiply by IR exponent
            if l == 0:
                bias_integrand = bias_integrand * expon * (-2./k/self.qint)**l
                bias_integrand -= bias_integrand[-1]
            else:
                bias_integrand = bias_integrand * expon * (-2./k/self.qint)**l
                                                                
            # do FFTLog
            ktemps, bias_fft = self.sph1.sph(l, bias_integrand)
            ret += interp1d(ktemps, bias_fft)(k)

        return 4*suppress*np.pi*ret + sn + k**2 * nu**2 * sn2 + k**4 * nu**4 * sn4
        
    def make_pknu_fixedbias(self, f, nu, bvec, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50,nmax=5):
    
        self.setup_rsd_facs(f,nu,nmax=nmax)
        
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)
        
        pknu= np.zeros(nk) # one column for ks

        for foo in range(nk):
            pknu[foo] = self.p_integral_fixedbias(kv[foo],bvec,nmax=nmax)
        
        return kv, pknu
        
    def make_pell_fixedbias(self, f, bvec, apar = 1, aperp = 1, ngauss=4, kv = None, kmin = 1e-2, kmax = 0.25, nk = 50,nmax=5):
        
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)

        pknutable = np.zeros((len(nus),nk)) # counterterms have distinct nu structure

        
        # To implement AP:
        # Calculate P(k,nu) at the true coordinates, given by
        # k_true = k_apfac * kobs
        # nu_true = nu * a_perp/a_par/fac
        # Note that the integration grid on the other hand is never observed
        
        for ii, nu in enumerate(nus_calc):
        
            fac = np.sqrt(1 + nu**2 * ((aperp/apar)**2-1))
            k_apfac = fac / aperp
            nu_true = nu * aperp/apar/fac
            vol_fac = apar * aperp**2
        
            self.setup_rsd_facs(f,nu_true)
            
            for jj, k in enumerate(kv):
                pknutable[ii,jj] = self.p_integral_fixedbias(k_apfac*k, bvec, nmax=nmax)
 
        
        pknutable[ngauss:,:] = np.flip(pknutable[0:ngauss],axis=0)
        

        p0k = 0.5 * np.sum((ws*L0)[:,None]*pknutable,axis=0) / vol_fac
        p2k = 2.5 * np.sum((ws*L2)[:,None]*pknutable,axis=0) / vol_fac
        p4k = 4.5 * np.sum((ws*L4)[:,None]*pknutable,axis=0) / vol_fac
        
        return kv, p0k, p2k, p4k
        
    def make_xiell_fixedbias(self, f, bvec, apar = 1, aperp = 1, ngauss=4, kmin = 1e-3, kmax = 0.8, nk = 100, nmax=5):

        kv, p0k, p2k, p4k = self.make_pell_fixedbias(f, bvec, apar=apar, aperp=aperp, ngauss=ngauss, kmin = kmin, kmax= kmax, nk = nk, nmax=nmax)
        
        damping = np.exp(-(self.kint/10)**2)
        p0int = loginterp(kv, p0k)(self.kint) * damping
        p2int = loginterp(kv, p2k)(self.kint) * damping
        p4int = loginterp(kv, p4k)(self.kint) * damping
        
        ss0, xi0 = self.sphr.sph(0,p0int)
        ss2, xi2 = self.sphr.sph(2,p2int); xi2 *= -1
        ss4, xi4 = self.sphr.sph(4,p4int)
        
        return (ss0, xi0), (ss2, xi2), (ss4, xi4)
      
