import numpy as np
from scipy.special import loggamma

import time

from Utils.loginterp import loginterp

class SphericalBesselTransform:

    def __init__(self, qs, L=15, low_ring=True, fourier=False):
    
        '''
        Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
        the untransformed coordinate, up to a given order L in bessel functions (j_l for l
        less than or equal to L. The point is to save time by evaluating the Mellin transforms
        u_m in advance.
        
        Does not use fftw as in spherical_bessel_transform_fftw.py, which makes it convenient
        to evaluate the generalized correlation functions in qfuncfft, as there aren't as many
        ffts as in LPT modules so time saved by fftw is minimal when accounting for the
        startup time of pyFFTW.
        
        Based on Yin Li's package mcfit (https://github.com/eelregit/mcfit)
        with the above modifications.
        
        '''

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2*np.pi**2)
        
        self.q = qs
        self.L = L
        
        self.Nx = len(qs)
        self.Delta = np.log(qs[-1]/qs[0])/(self.Nx-1)

        self.N = 2**(int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.pads = np.zeros( (self.N-self.Nx)//2  )
        self.pad_iis = np.arange(self.Npad - self.Npad//2, self.N - self.Npad//2)
        
        # Set up the FFTLog kernels u_m up to, but not including, L
        ms = np.arange(0, self.N//2+1)
        self.ydict = {}; self.udict = {}; self.qdict= {}
        
        if low_ring:
            for ll in range(L):
                q = max(0, 1.5 - ll)
                lnxy = self.Delta/np.pi * np.angle(self.UK(ll,q+1j*np.pi/self.Delta)) #ln(xmin*ymax)
                ys = np.exp( lnxy - self.Delta) * qs/ (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) \
                        * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
        
        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(L):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q



    
    def sph(self, nu, fq):
        '''
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        '''
        q = self.qdict[nu]; y = self.ydict[nu]
        f = np.concatenate( (self.pads,self.q**(3-q)*fq,self.pads) )
        
        fks = np.fft.rfft(f)
        gks = self.udict[nu] * fks
        gs = np.fft.hfft(gks) / self.N

        return y, y**(-q) * gs[self.pad_iis]
    
    

    def UK(self, nu, z):
        '''
        The Mellin transform of the spherical bessel transform.
        '''
        return self.sqrtpi * np.exp(np.log(2)*(z-2) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))

    def update_tilt(self,nu,tilt):
        '''
        Update the tilt for a particular nu. Assume low ring coordinates.
        '''
        q = tilt; ll = nu
        
        ms = np.arange(0, self.N//2+1)
        lnxy = self.Delta/np.pi * np.angle(self.UK(ll,q+1j*np.pi/self.Delta)) #ln(xmin*ymax)
        ys = np.exp( lnxy - self.Delta) * self.q/ (self.q[0]*self.q[-1])
        us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms) \
                * np.exp(-2j * np.pi * lnxy / self.N / self.Delta * ms)
                
        self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
