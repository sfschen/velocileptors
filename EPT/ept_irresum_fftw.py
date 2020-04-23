import numpy as np

from EPT.ept_fftw import EPT

class REPT:

    '''
    Class to compute IR-resummed velocity moments and RSD using the moment expansion appraoch in EPT.
    
    Based on the EPT class.
    
    '''
    
    def __init__(self, k, p, pnw, *args, rbao = 110, kmin = 1e-2, kmax = 0.5, nk = 100, **kw):
        
        self.nk, self.kmin, self.kmax = nk, kmin, kmax
        self.rbao = rbao
        
        self.ept = EPT( k, p, kmin=kmin, kmax=kmax, **kw)
        self.ept_nw = EPT( k, pnw, kmin=kmin, kmax=kmax, **kw)
        self.beyond_gauss = self.ept.beyond_gauss
        
        self.kv = self.ept.kv
        self.sigma_squared_bao = np.interp(self.rbao, self.ept_nw.qint, self.ept_nw.Xlin + self.ept_nw.Ylin/3.)
        self.damp_exp = - 0.5 * self.kv**2 * self.sigma_squared_bao
        self.damp_fac = np.exp(self.damp_exp)
        
        
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
    # In this case the parameters alpha_g1, alpha_g3, alpha_k2, stoch_k0 are not used.
    # Otherwise gives the full moment expansion expression up to one-loop order.
    
    def combine_bias_terms_pkrsd(self,nu,f,\
                                 b1,b2,bs,b3,\
                                 alpha,alphav,alpha_s0,alpha_s2,alpha_g1,alpha_g3,alpha_k2,\
                                 sn,sv,sigma0,stoch_k0,\
                                 ct3 = 0, beyond_gauss=False):
        
        kv = self.pktable[:,0]
        nu2 = nu**2
        
        pk = self.combine_bias_terms_pk(b1,b2,bs,b3,alpha,sn)
        vk = self.combine_bias_terms_vk(b1,b2,bs,b3,alphav,sv)
        s0k, s2k = self.combine_bias_terms_sk(b1,b2,bs,b3,alpha_s0,alpha_s2,sigma0)
        
        ret = pk - f*kv*nu2*vk - 0.5*f**2*kv**2*nu2 * (s0k + 0.5*s2k*(3*nu2-1) )
        
        if beyond_gauss:
            g1k, g3k = self.combine_bias_terms_gk(b1,b2,bs,b3,alpha_g1,alpha_g3)
            k0k, k2k, k4k = self.combine_bias_terms_kk(b1,b2,bs,b3,alpha_k2,stoc_k0)
            ret += 1./6 * f**3 * (kv*nu)**3 * (g1k * nu + g3k * nu**3)\
                   + 1./24 * f**4 * (kv*nu)**4 * (k0k + k2k * nu2 + k4k * nu2**2)
        else:
            ret += 1./6 * ct3 * kv**2 * nu**4 * self.pktable[:,-1]
        
        return kv, ret
                
                
    def combine_bias_terms_pkrsd_reduced(self,nu,f,b1,b2,bs,b3,alpha,alpha2,alpha4,alpha6,sn,s2,sn4,gaussian=True):
    
        return self.combine_bias_terms_pkrsd(nu,f,b1,b2,bs,b3,alpha,alpha2,0,alpha4,0,alpha_6,0,sn,s2,0,sn4, gaussian=gaussian)
