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


The core rsd modules are distributed into two directories, one for
the LPT theory and one for the EPT theory (plus supporting routines
in the Utils directory). Running the EPT version of the above simply
requires substituting the "LPT" for "EPT" in the import statement.
See the READMEs in those directories for more information.

For the most common case of computing the redshift-space power spectrum
one can use a a _reduced set_ of parameters and beyond_gauss=False:

pars = [b1, b2, bs, b3] +  [alpha0, alpha2, alpha4] +  [sn, sn2]

where

(1) b1, b2, bs, b3:  the bias parameters up to cubic order

(2) alpha0, alpha2, alpha4: counter terms of the form mu^n.

(3) sn, sn2: stochastic contributions to P_real(k), sigma^2
    [e.g. shot-noise, finger-of-god dispersion and kappa].


To include higher order terms in the moment expansion set beyond_gauss=True.
In this case you need to additionally specify an alpha6 counterterm and
an sn4 stochastic term.

-------

If you additionally want access to the velocity statistics, then the
full set of parameters is

pars = [b1, b2, bs, b3] +  [alpha, alpha_v, alpha_s0, alpha_s2, alpha_g1, alpha_g3, alpha_k2] +  [sn, sv, s0, s4]

where the parameters are:

(1) b1, b2, bs, b3: the bias parameters up to cubic order

(2) alpha, alpha_v, alpha_s0, alpha_s2, alpha_g1, alpha_g3, alpha_k2: the one-loop counterterms for each velocity component

(3) sn, sv, s0, s4: the stochastic contributions to the velocities

Again, for most practical purposes one can set beyond_gauss=False,
in which case the code only uses up to sigma(k), and uses a counterterm
ansatz for the third and fourth moments.
In this case the parameters alpha_g1, alpha_g3, alpha_k2, and s4 are not used.

More details can be found in Chen, Vlah & White (2020).

---

This code is related to the configuration-space based code
https://github.com/martinjameswhite/CLEFT_GSM.
