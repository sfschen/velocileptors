import numpy as np
import pyfftw
import pickle
from scipy.special import loggamma

import time

from loginterp import loginterp

# Class to perform spherical bessel transforms via FFTLog for a given set of qs, ie.
# the untransformed coordinate, up to a given order L in bessel functions (j_l for l
# less than or equal to L. The point is to save time by evaluating the Mellin transforms
# u_m in advance.

# Uses pyfftw, which can perform multiple (ncol) Fourier transforms at once, one for each bias contribution.

class SphericalBesselTransform:

    def __init__(self, qs, L=15, ncol = 1, low_ring=True, fourier=False, threads=1,
                 import_wisdom=False, wisdom_file='./fftw_wisdom.npy'):

        # numerical factor of sqrt(pi) in the Mellin transform
        # if doing integral in fourier space get in addition a factor of 2 pi / (2pi)^3
        if not fourier:
            self.sqrtpi = np.sqrt(np.pi)
        else:
            self.sqrtpi = np.sqrt(np.pi) / (2*np.pi**2)
        
        self.q = qs
        self.L = L
        self.ncol = ncol
        
        self.Nx = len(qs)
        self.Delta = np.log(qs[-1]/qs[0])/(self.Nx-1)
        
        # zero pad the arrays to the preferred length format for ffts, 2^N
        self.N = 2**(int(np.ceil(np.log2(self.Nx))) + 1)
        self.Npad = self.N - self.Nx
        self.ii_l = self.Npad - self.Npad//2 # left and right indices sandwiching the padding
        self.ii_r = self.N - self.Npad//2
        
        # Set up FFTW objects:
        if import_wisdom:
            pyfftw.import_wisdom(tuple(np.load(wisdom_file)))
        
        self.fks = pyfftw.empty_aligned((self.ncol,self.N//2 + 1), dtype='complex128')
        self.fs  = pyfftw.empty_aligned((self.ncol,self.N), dtype='float64')
        
        pyfftw.config.NUM_THREADS = threads
        self.fft_object = pyfftw.FFTW(self.fs, self.fks, direction='FFTW_FORWARD',threads=threads)
        self.ifft_object = pyfftw.FFTW(self.fks, self.fs, direction='FFTW_BACKWARD',threads=threads)
        
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
                us[self.N//2] = us[self.N//2].real # manually impose low ring

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q
        
        else:
            # if not low ring then just set x_min * y_max = 1
            for ll in range(L):
                q = max(0, 1.5 - ll)
                ys = np.exp(-self.Delta) * qs / (qs[0]*qs[-1])
                us = self.UK(ll, q + 2j * np.pi / self.N / self.Delta * ms)
                us[self.N//2] = us[self.N//2].real # manually impose low ring

                self.ydict[ll] = ys; self.udict[ll] = us; self.qdict[ll] = q


    def export_wisdom(self, wisdom_file='./fftw_wisdom.npy'):
        np.save(wisdom_file, pyfftw.export_wisdom())
    
    def sph(self, nu, fq):
        '''
        The workhorse of the class. Spherical Hankel Transforms fq on coordinates self.q.
        '''
        q = self.qdict[nu]; y = self.ydict[nu]
        self.fs[:,self.Npad - self.Npad//2 : self.N - self.Npad//2] = fq * self.q**(3-q)
        
        fks = self.fft_object()
        self.fks[:] = np.conj(fks * self.udict[nu])
        gs = self.ifft_object()

        return y, gs[:,self.ii_l:self.ii_r] * y**(-q)
    
    

    def UK(self, nu, z):
        '''
        The Mellin transform of the spherical bessel transform.
        '''
        return self.sqrtpi * np.exp(np.log(2)*(z-2) + loggamma(0.5*(nu+z)) - loggamma(0.5*(3+nu-z)))

