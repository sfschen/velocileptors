# velocileptors

Velocity-based perturbation theory expansions of redshift-space distortions and velocity statistics, available in both Lagrangian (LPT) and Eulerian (EPT) formulations.

This code computes the real- and redshift-space power spectra and correlation functions of biased tracers using 1-loop perturbation theory (with effective field theory counter terms and up to cubic biasing), as well as the real-space pairwise velocity moments.

---

## Installation

```
python3 -m pip install -v git+https://github.com/sfschen/velocileptors
```

### Dependencies

- `numpy`
- `scipy`
- `pyFFTW` (Python wrapper for FFTW) — https://hgomersall.github.io/pyFFTW/

> pyFFTW is pip and conda installable. The conda-forge channel is recommended as the most reliable source.

---

## Quick Start

For most situations, computing the power spectrum wedges or multipoles is as simple as:

```python
from velocileptors.LPT.moment_expansion_fftw import MomentExpansion

mome        = MomentExpansion(klin, pklin, threads=nthreads)
kw, pkw     = mome.compute_redshift_space_power_at_mu(pars, f, mu, reduced=True)
kl,p0,p2,p4 = mome.compute_redshift_space_power_multipoles(pars, f, reduced=True)
```

> To use the EPT version, substitute `"LPT"` for `"EPT"` in the import statement. For added robustness, supply a cosmology-conscious no-wiggle power spectrum via the `pnw` keyword.

---

## Example Notebooks & Scripts

| Resource | Description |
|---|---|
| `Moment Expansion Example.ipynb` | Reproduces the plots from the paper |
| `Fourier Streaming Model Example.ipynb` | Steps for the Fourier Streaming Model |
| `Gaussian Streaming Model Example.ipynb` | Produces correlation function multipoles |
| `lpt_examples.py` | Most common LPT use-cases |
| LPT & EPT notebooks | Detailed power spectrum calculations |

Most users will want to use either the `lpt_rsd` or `ept_fullresum` models, described in the **Direct LPT RSD Examples** and **EPT Examples** notebooks.

---

## Module Structure

The core RSD modules are split into two directories:

- **`LPT/`** — Lagrangian perturbation theory modules
- **`EPT/`** — Eulerian perturbation theory modules
- **`Utils/`** — Supporting routines

---

## Parameters

### Reduced Parameter Set (most common use case)

For redshift-space power spectrum calculations, use `beyond_gauss=False`:

```
pars = [b1, b2, bs, b3] + [alpha0, alpha2, alpha4] + [sn, sn2]
```

| Parameter | Description |
|---|---|
| `b1, b2, bs, b3` | Bias parameters up to cubic order |
| `alpha0, alpha2, alpha4` | Counter terms of the form μⁿ |
| `sn, sn2` | Stochastic contributions to P_real(k) and σ² (e.g. shot-noise, finger-of-god dispersion, kappa) |

> Set `beyond_gauss=True` to include higher-order moment expansion terms. This requires additionally specifying an `alpha6` counterterm and an `sn4` stochastic term.
> AP parameters can be set via the `apar` and `aperp` keywords.

### Full Parameter Set (including velocity statistics)

```
pars = [b1, b2, bs, b3] + [alpha, alpha_v, alpha_s0, alpha_s2, alpha_g1, alpha_g3, alpha_k2] + [sn, sv, s0, s4]
```

| Parameter | Description |
|---|---|
| `b1, b2, bs, b3` | Bias parameters up to cubic order |
| `alpha, alpha_v, alpha_s0, alpha_s2, alpha_g1, alpha_g3, alpha_k2` | One-loop counterterms for each velocity component |
| `sn, sv, s0, s4` | Stochastic contributions to the velocities |

> With `beyond_gauss=False`, the code uses up to σ(k) and a counterterm ansatz for the third and fourth moments. In this case, `alpha_g1`, `alpha_g3`, `alpha_k2`, and `s4` are not used.

---

## Direct Expansion Modules

Two additional "direct expansion" modules are available in both LPT and EPT:

- **`LPT.lpt_rsd_fftw`** — described in [arXiv:2012.04636](https://arxiv.org/abs/2012.04636)
- **`EPT.ept_fullresum_fftw`** — described in [arXiv:2005.00523](https://arxiv.org/abs/2005.00523)

Both take the full bias vector:

```
pars = [b1, b2, bs, b3] + [alpha0, alpha2, alpha4, alpha6] + [sn, sn2, sn4]
```

### Note on the LPT Module

The LPT module must be called differently from the others because its angular dependence requires recomputing PT integrals at each μ. Example for multipoles:

```python
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
lpt = LPT_RSD(klin, plin, kIR=0.2)

lpt.make_pltable(f, nmax=4, apar=1, aperp=1)
kl, p0, p2, p4 = lpt.combine_bias_terms_pkell(pars)
```

For wedges P(k,μ) and further examples, see the example notebooks.

---

## Validation

velocileptors was entered into the [blind mock challenge](https://www2.yukawa.kyoto-u.ac.jp/~takahiro.nishimichi/data/PTchallenge/) described [here](https://arxiv.org/abs/2003.08277). Results are summarized below, where the shaded regions are errors for a 5 (Gpc/h)³ volume with the same signal.

![PT challenge](https://github.com/sfschen/velocileptors/blob/master/param_plot_desi_vol.png?raw=true)

---

## References

- Chen, Vlah & White (2020) — https://arxiv.org/abs/2005.00523
- Chen, Vlah, Castorina & White (2020) — https://arxiv.org/abs/2012.04636

---

## Related & Acknowledgements

- **Related code (configuration-space):** [CLEFT_GSM](https://github.com/martinjameswhite/CLEFT_GSM)
- **Based on earlier LPT code by Chirag Modi:** [CLEFT](https://github.com/modichirag/CLEFT), which used the [mcfit](https://github.com/eelregit/mcfit) class.

> The main modification over the original is that Mellin transform kernels used in FFTLog are saved to reduce computation time, since they are more expensive to evaluate than the FFTs themselves.

Thanks to **Arnaud De Mattia** for help debugging features involving massive neutrinos.