# velocileptors
Velocity-based perturbation theory (both Lagrangian (LPT) and Eulerian (EPT)
formulations) expansions of redshift-space distortions and
velocity statistics.

This code computes the real- and redshift-space power spectra and
correlation functions of biased tracers using 1-loop perturbation
theory (with effective field theory counter terms and up to cubic
biasing) as well as the real-space pairwise velocity moments.

The code requires numpy, scipy and pyFFTW (the python wrapper for FFTW):

https://hgomersall.github.io/pyFFTW/

to run. Note that pyFFTW is pip and conda installable and available
from conda-forge (we have found the conda-forge channel to be the
most reliable).

An example calculation to reproduce the plots in the paper is given
in "Moment Expansion Example.ipynb".
A short notebook detailing the same steps for the Fourier Streaming Model
is given in "Fourier Streaming Model Example.ipynb".
Also included is "Gaussian Streaming Model Example.ipynb" that runs through
how to produce the correlation function multipoles.
An example of the most common use-cases is given in "lpt_examples.py".

For most situations computing the power spectrum wedges or multipoles
is as simple as:

```
from LPT.moment_expansion_fftw import MomentExpansion

mome        = MomentExpansion(klin,pklin,threads=nthreads)
kw,pkw      = mome.compute_redshift_space_power_at_mu(pars,f,mu,reduced=True)
kl,p0,p2,p4 = mome.compute_redshift_space_power_multipoles(pars,f,reduced=True)
```


The core rsd modules are distributed into two directories, one for the LPT theory and
one for the EPT theory (plus supporting routines in the Utils directory).  See the READMEs
in those directories for more information.

For the most common case of computing the redshift-space power spectrum
one can use a a _reduced set_ of parameters:

pars = [b1, b2, bs, b3] +  [alpha0, alpha2, alpha4] +  [sn, sn2]

where

(1) b1, b2, bs, b3:  the bias parameters up to cubic order

(2) alpha0, alpha2, alpha4: counter terms for ell=0, 2 and 4.

(3) sn, sn2: stochastic contributions to P_real(k) and sigma^2
    [e.g. shot-noise and finger-of-god dispersion].


If you additionally want access to the velocity statistics, then the
full set of parameters is

pars = [b1, b2, bs, b3] +  [alpha, alpha_v, alpha_s0, alpha_s2] +  [sn, sv, s0]

where the parameters are:

(1) b1, b2, bs, b3: the bias parameters up to cubic order

(2) alpha, alpha_v, alpha_s0, alpha_s2: the one-loop counterterms for each velocity component

(3) sn, sv, s0: the stochastic contributions to the velocities

More details can be found in Chen, Vlah & White (2020).


This code is related to the configuration-space based code https://github.com/martinjameswhite/CLEFT_GSM.
