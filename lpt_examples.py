#!/usr/bin/env python3
#
# An example of how to call the codes to compute the
# redshift-space power spectrum of biased tracers.
# The key "code" is MomentExpansion.
#
import numpy as np
import os
from   LPT.moment_expansion_fftw import MomentExpansion



def set_up():
    """Set everything up given Plinear(k) -- this is the
       most time consuming part but only has to be done once.
       It is faster if you have a wisdom file already and
       pass the file name."""
    nthr       = int(os.getenv('OMP_NUM_THREADS','1'))
    z,D        = 0.8,0.6819
    klin,plin  = np.loadtxt("pk.dat",unpack=True)
    plin      *= D**2
    mome = MomentExpansion(klin,plin,threads=nthr,\
             cutoff=10,extrap_min=-4,extrap_max=3,jn=10)
    return(mome)
    #




    


if __name__=="__main__":
    # Set up the instance.
    mome = set_up()
    # Set up some fiducial values for the parameters -- we'll use the
    # examples from the paper and we'll used the "reduced" parameter set:
    # pars: the biases and counter terms.
    #    b1,b2,bs, b3: linear, quadratic & cubic parameters
    #    alpha0,alpha2,alpha4: counterterms
    #    sn,s0: stochastic contributions to P(k) and sigma^2.
    z,f    = 0.80,0.8076
    biases = [0.70,0.5,-0.3,0.0]
    cterms = [10.0,20.,-60.]
    stoch  = [1800.,-1000.]
    pars   = biases + cterms + stoch
    # Compute the wedges, here we'll just to a single mu.  Note if we
    # just wanted the real-space power spectrum we could pass mu=0 to
    # compute_redshift_space_power_at_mu.  Here we'll do mu=0.5:
    mu     = 0.5
    kw,pkw = mome.compute_redshift_space_power_at_mu(pars,f,mu,reduced=True)
    print("First few k bins for mu=",mu)
    for k,p in zip(kw[:10],pkw[:10]):
        print("{:12.4e} {:12.4e}".format(k,p))
    #
    # Compute the multipoles.
    kl,p0,p2,p4 = mome.compute_redshift_space_power_multipoles(pars,f,reduced=True)
    print("\n\nFirst few k bins for multipoles")
    for k,mono,quad,hexa in zip(kl[:10],p0[:10],p2[:10],p4[:10]):
        print("{:12.4e} {:12.4e} {:12.4e} {:12.4e}".format(k,mono,quad,hexa))
