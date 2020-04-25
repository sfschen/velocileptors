# velocileptors

This is the EPT branch of the velocileptors code.

The main workhorses are cleft_kexpanded_fftw.py and velocity_moments_kexpanded_fftw.py.
These compute the EPT power spectra contributions at one-loop using Hankel transformed
inspired by LPT, which are then transformed linearly into EPT in ept_fftw.py. The main
difference with LPT is that the Hankel transforms can be done for all k at once instead
of for each k point.

However, the resulting expressions need to be IR-resummed. This is done in two classes:

1. moment_expansion_fftw.py: computes the IR-resummed velocity moments. These are then 
added together to form the power spectrum in compute_redshift_space_power_at_mu and
compute_redshift_space_multipoles. The velocity moments are accesible using the functions
"combine_bias_terms_*", where * = pk, vk, sk, gk, kk for the 0th to 4th moments.

2. ept_fullresum_fftw.py: this directly computes the IR-resummed one loop EPT power
spectrum without going through the velocities.

The basic call is:
```
from moment_expansion_fftw import MomentExpansion

mome        = MomentExpansion(klin,pklin, pnw=pnw, threads=nthreads)
kw,pkw      = mome.compute_redshift_space_power_at_mu(pars,f,mu,reduced=True)
kl,p0,p2,p4 = mome.compute_redshift_space_power_multipoles(pars,f,reduced=True)
```
If no no-wiggle spectrum is given, the code will calculate one on-the-fly using a Savitsky-Golay filter,
but we have found that calculations are more robust when using no-wiggle spectra that have been calculated
with some cosmology-dependence smoothed out (e.g. Eisenstein and Hu), which we leave to the user.

Examples are given in the main directory under "EPT Examples.ipynb."
