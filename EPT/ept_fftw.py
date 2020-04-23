import numpy as np

from Utils.loginterp import loginterp
from EPT.velocity_moments_kexpanded_fftw import KEVelocityMoments

class EPT(KEVelocityMoments):

    '''
    Class based on velocity_moments_kexpanded_fftw to compute ept velocity moments.
    
    Computes one-loop terms using Hankel transforms inspired by LPT, then transforms them linearly.
    
    Not IR-resummed.
    
    '''

    def __init__(self, *args, kmin = 1e-2, kmax = 0.5, nk = 100, **kw):
    
        # Initialize the velocity class
        KEVelocityMoments.__init__(self, *args, **kw)
        
        self.nk, self.kmin, self.kmax = nk, kmin, kmax
        self.make_tables(kmin=kmin, kmax=kmax, nk = nk)
        self.kv = self.pktable[:,0]
        self.plin = loginterp(self.k, self.p)(self.kv)

        # Now convert to EPT table
        self.convert_ptable()
        self.convert_vtable()
        self.convert_stable()
        self.convert_gtable()
        
    
    def convert_ptable(self):
        
        # Make SPT spectra out of linear combinations of LPT spectra
        
        self.pktable_ept = np.zeros( (self.nk, 9)  )
        
        pktable = self.pktable
        
        self.pktable_ept[:,0] = pktable[:,0]
        self.pktable_ept[:,1] = pktable[:,1] # b1sq
        self.pktable_ept[:,2] = pktable[:,5] - 16./21 * pktable[:,6] + 2./7 * pktable[:,9] # b1b2
        self.pktable_ept[:,3] = pktable[:,8] + 4./7 * pktable[:,10] - 8./21 * pktable[:,9] # b1bs
        self.pktable_ept[:,4] = pktable[:,6] # b2sq
        self.pktable_ept[:,5] = pktable[:,9] # b2bs
        self.pktable_ept[:,6] = pktable[:,10] # bssq
        self.pktable_ept[:,7] = 2*pktable[:,3] - pktable[:,2] - 8./21*pktable[:,5] + 2./7*pktable[:,8] # b1b3
        self.pktable_ept[:,8] = self.kv**2 * self.plin # ct
        
        # linear theory table
        self.pktable_ept_linear = np.zeros( (self.nk, 9)  )
        pktable = self.pktable_linear
        
        self.pktable_ept_linear[:,0] = pktable[:,0]
        self.pktable_ept_linear[:,1] = pktable[:,1] # b1sq

        
        
    def convert_vtable(self):
    
        self.vktable_ept = np.zeros( (self.nk, 9)  )
        
        vktable = self.vktable
        
        self.vktable_ept[:,0] = vktable[:,0]
        self.vktable_ept[:,1] = vktable[:,1] - vktable[:,3] + 8./21*vktable[:,5] - 2./7*vktable[:,8] # b1
        self.vktable_ept[:,2] = vktable[:,3] - 8./21*vktable[:,5] + 2./7*vktable[:,8] # b1sq
        self.vktable_ept[:,3] = vktable[:,4] - vktable[:,5] # b2
        self.vktable_ept[:,4] = vktable[:,5] # b1b2
        self.vktable_ept[:,5] = vktable[:,7] - vktable[:,8]# bs
        self.vktable_ept[:,6] = vktable[:,8] # b1bs
        self.vktable_ept[:,7] = vktable[:,2] - vktable[:,1] - vktable[:,3] \
        - 8./21*(vktable[:,4] - vktable[:,5]) + 2./7 * (vktable[:,7] - vktable[:,8]) # b3
        self.vktable_ept[:,8] = self.kv * self.plin # ct

        # linear theory table
        self.vktable_ept_linear = np.zeros( (self.nk, 9)  )
        vktable = self.vktable_linear
        
        self.vktable_ept_linear[:,0] = vktable[:,0]
        self.vktable_ept_linear[:,1] = vktable[:,1] - vktable[:,3] + 8./21*vktable[:,5] - 2./7*vktable[:,8] # b1

    def convert_stable(self):
    
        self.s0ktable_ept = np.zeros( (self.nk, 7))
        self.s2ktable_ept = np.zeros( (self.nk, 7))

        s0 = self.s0; s2 = self.s2
        
        self.s0ktable_ept[:,0] = s0[:,0]
        self.s0ktable_ept[:,1] = s0[:,1] - s0[:,2] + s0[:,3] + 8./21*s0[:,4] - 2./7*s0[:,7]  # 1
        self.s0ktable_ept[:,2] = s0[:,2] - 2*s0[:,3] - 8./21*s0[:,4] + 2./7*s0[:,7] # b1
        self.s0ktable_ept[:,3] = s0[:,3] # b1sq
        self.s0ktable_ept[:,4] = s0[:,4] # b2
        self.s0ktable_ept[:,5] = s0[:,7] # bs
        self.s0ktable_ept[:,6] = self.plin # ct
        
        self.s2ktable_ept[:,0] = s2[:,0]
        self.s2ktable_ept[:,1] = s2[:,1] - s2[:,2] + s2[:,3] + 8./21*s2[:,4] - 2./7*s2[:,7]  # 1
        self.s2ktable_ept[:,2] = s2[:,2] - 2*s2[:,3] - 8./21*s2[:,4] + 2./7*s2[:,7] # b1
        self.s2ktable_ept[:,3] = s2[:,3] # b1sq
        self.s2ktable_ept[:,4] = s2[:,4] # b2
        self.s2ktable_ept[:,5] = s2[:,7] # bs
        self.s2ktable_ept[:,6] = self.plin # ct
        
        # linear theory table
        self.s0ktable_ept_linear = np.zeros( (self.nk, 7))
        self.s2ktable_ept_linear = np.zeros( (self.nk, 7))

        s0 = self.s0_linear; s2 = self.s2_linear
        
        self.s0ktable_ept_linear[:,0] = s0[:,0]
        self.s0ktable_ept_linear[:,1] = s0[:,1] - s0[:,2] + s0[:,3] + 8./21*s0[:,4] - 2./7*s0[:,7]  # 1
        
        self.s2ktable_ept_linear[:,0] = s2[:,0]
        self.s2ktable_ept_linear[:,1] = s2[:,1] - s2[:,2] + s2[:,3] + 8./21*s2[:,4] - 2./7*s2[:,7]  # 1
        
    
    def convert_gtable(self):
        
        self.g1ktable_ept = np.zeros( (self.nk, 4) )
        self.g3ktable_ept = np.zeros( (self.nk, 4) )
        
        self.g1ktable_ept[:,0] = self.g1[:,0]
        self.g1ktable_ept[:,1] = self.g1[:,1] - self.g1[:,2] # 1
        self.g1ktable_ept[:,2] = self.g1[:,2] # b1
        self.g1ktable_ept[:,3] = self.plin/self.kv # ct
        
        self.g3ktable_ept[:,0] = self.g3[:,0]
        self.g3ktable_ept[:,1] = self.g3[:,1] - self.g3[:,2]
        self.g3ktable_ept[:,2] = self.g3[:,2]
        self.g3ktable_ept[:,3] = self.plin/self.kv
        
    
    def combine_bias_terms_pk(self,b1,b2,bs,b3,alpha,sn):
        
        return b1**2 * self.pktable_ept[:,1] + b1*b2 * self.pktable_ept[:,2] + b1*bs * self.pktable_ept[:,3] \
                + b2**2 * self.pktable_ept[:,4] + b2*bs * self.pktable_ept[:,5] + bs**2 * self.pktable_ept[:,6] \
                + b1*b3 * self.pktable_ept[:,7] + alpha*self.pktable_ept[:,8] + sn
                
    def combine_bias_terms_vk(self,b1,b2,bs,b3,alphav, sv):
        
        return b1 * self.vktable_ept[:,1] + b1**2 * self.vktable_ept[:,2] + b2 * self.vktable_ept[:,3] \
              +b1*b2*self.vktable_ept[:,4] + bs*self.vktable_ept[:,5] + b1*bs * self.vktable_ept[:,6] \
              +b3*self.vktable_ept[:,7] + alphav * self.vktable_ept[:,8] + sv * self.vktable_ept[:,0]
              
    def combine_bias_terms_sk(self,b1,b2,bs,b3,alpha0,alpha2,sigma0):
        
        s0 = self.s0ktable_ept[:,1] + b1 * self.s0ktable_ept[:,2] + b1**2 * self.s0ktable_ept[:,3] \
            + b2 * self.s0ktable_ept[:,4] + bs * self.s0ktable_ept[:,5] + alpha0*self.s0ktable_ept[:,6] + sigma0
            
        s2 = self.s2ktable_ept[:,1] + b1 * self.s2ktable_ept[:,2] + b1**2 * self.s2ktable_ept[:,3] \
        + b2 * self.s2ktable_ept[:,4] + bs * self.s2ktable_ept[:,5] + alpha2*self.s2ktable_ept[:,6]
        
        return s0, s2
        
        
    def combine_bias_terms_gk(self,b1,b2,bs,b3,alpha1,alpha3):
    
        g1 = self.g1ktable_ept[:,1] + b1*self.g1ktable_ept[:,2] + alpha1 * self.g1ktable_ept[:,-1]
        g3 = self.g3ktable_ept[:,1] + b1*self.g3ktable_ept[:,2] + alpha3 * self.g1ktable_ept[:,-1]
        
        return g1, g3
        
    def combine_bias_terms_kk(self,b1,b2,bs,b3):
    
        # These are matter velocity only so don't need to be transformed
        
        k0 = self.k0
        k2 = self.k2
        k4 = self.k4
        
        return k0, k2, k4
        
        
    def combine_bias_terms_pkrsd(self,nu,f,b1,b2,bs,b3,alpha,alphav,alpha0,alpha2,sn,sv,sigma0, gaussian = True):
        
        kv = self.pktable_ept[:,0]
        nu2 = nu**2
        
        pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha,sn)
        vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alphav,sv)
        s0k, s2k = self.combine_bias_terms_sk(b1,b2,bs,b3,alpha0,alpha2,sigma0)
        
        ret = pk - f*kv*nu2*vk - 0.5*f**2*kv**2*nu2 * (s0k + 0.5*s2k*(3*nu2-1) )
        
        if gaussian == False:
            g1k, g3k = self.combine_bias_terms_gk(b1,b2,bs,b3)
            k0k, k2k, k4k = self.combine_bias_terms_kk(b1,b2,bs,b3)
            ret += 1./6 * f**3 * (kv*nu)**3 * (g1k * nu + g3k * nu**3)\
                   + 1./24 * f**4 * (kv*nu)**4 * (k0k + k2k * nu2 + k4k * nu2**2)
        
        return kv, ret
                
                
    def combine_bias_terms_pkrsd_reduced(self,nu,f,b1,b2,bs,b3,alpha,alpha2,alpha4,sn,s2,gaussian=True):
    
        return self.combine_bias_terms_pkrsd(nu,f,b1,b2,bs,b3,alpha,alpha2,0,alpha4,sn,s2,0,gaussian=gaussian)
    
