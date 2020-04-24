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

Examples are given in the main directory under "EPT Examples.ipynb."
