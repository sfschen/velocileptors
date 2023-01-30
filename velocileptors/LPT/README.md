# velocileptors/LPT

The Lagrangian PT branch of the velocileptors code.

This code computes the real- and redshift-space power spectra and
correlation functions of biased tracers using 1-loop perturbation
theory (with effective field theory counter terms and up to cubic
biasing) as well as the real-space pairwise velocity moments.

For most situations computing the power spectrum wedges or multipoles
is as simple as:

```
from moment_expansion_fftw import MomentExpansion

mome        = MomentExpansion(klin,pklin,threads=nthreads)
kw,pkw      = mome.compute_redshift_space_power_at_mu(pars,f,mu,reduced=True)
kl,p0,p2,p4 = mome.compute_redshift_space_power_multipoles(pars,f,reduced=True)
```

The core rsd modules are:

(1) moment_expansion_fftw.py: this computes real- and redshift-space power
spectra using the moment expansion approach.

(2) gaussian_streaming_model_fftw.py: this computes real- and redshift-space
correlation functions using the Gaussian streaming model.

(3) fourier_streaming_model_fftw.py: this computes redshift-space
power spectra using the Fourier streaming model. 

(4) lpt_rsd_fftw.py: this computes the power spectrum in redshift space via direct Lagrangian expansion.

All the models except (4) use velocity spectra from velocity_moments_fftw.py,
which itself inherits the real-space power spectrum module cleft_fftw.py. The model
in (4) is described in https://arxiv.org/abs/2012.04636 .



For the most common case of computing the redshift-space power spectrum
one can use a a _reduced set_ of parameters and beyond_gauss=False:

pars = [b1,b2,bs,b3] +  [alpha0,alpha2,alpha4] +  [sn,sn2]

where

(1) b1, b2, bs, b3:  the bias parameters up to cubic order

(2) alpha0,alpha2,alpha4: counter terms of the form mu^n.

(3) sn, sn2: stochastic contributions to P_real(k) and sigma^2
    [e.g. shot-noise and finger-of-god dispersion].
 
---

If the beyond_gauss setting is True in (1)-(3) or if using (4), then the parameters are instead

pars = [b1,b2,bs,b3] +  [alpha0,alpha2,alpha4,alpha6] +  [sn,sn2,sn4]

However, for most purposes the extra parameters can be set to zero with little effect.

---

If you additionally want access to the velocity statistics, then the
full set of parameters is

pars = [b1, b2, bs, b3] +  [alpha, alpha_v, alpha_s0, alpha_s2] +  [sn, sv, s0]

where the parameters are:

(1) b1, b2, bs, b3: the bias parameters up to cubic order

(2) alpha, alpha_v, alpha_s0, alpha_s2: the one-loop counterterms
for each velocity component

(3) sn, sv, s0: the stochastic contributions to the velocities

Since the expansion for realistic samples is saturated at the third moment,
for most practical purposes one can set beyond_gauss = False, in which
case the code only uses up to sigma(k), and uses a counterterm ansatz
for the third and fourth moments. In this case the parameters
alpha_g1, alpha_g3, alpha_k2, and s4 are not used and can safely be set
to zero.

More details can be found in Chen, Vlah & White (2020).

---

This code is related to the configuration-space based code https://github.com/martinjameswhite/CLEFT_GSM.

