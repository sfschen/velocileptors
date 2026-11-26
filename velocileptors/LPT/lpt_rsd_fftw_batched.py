"""Vectorized make_pltable for LPT_RSD.

The stock `LPT_RSD.make_pltable` runs its 49-k inner loop serially and calls
`p_integrals` once per k. Each `p_integrals` call does numpy arithmetic on
`(jn, num_power_components, N) ≈ (5, 13, 256)` tensors and a handful of tiny
pyfftw FFTs — too small for either numpy BLAS or pyfftw threading to pay off.

`LPT_RSD_Batched` replaces that with a batched computation: `p_integrals_batched`
evaluates all k values at once in one big tensor pass. The pyfftw transform is
run via a single `SphericalBesselTransform` whose `ncol = nk * num_power_components`,
so the FFT engine sees a ~50× larger job that actually benefits from
`OMP_NUM_THREADS`.

Public API matches the parent: the only new method a caller might notice is
`p_integrals_batched`. Output tables (`p0/2/4ktable`, `pres0/2/4ktable`,
`pknutable`, `presknutable`) are bit-close to the serial version.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD, _interp_k_many
from velocileptors.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform


_POOL = {}


def _get_pool(n):
    # one shared pool per size, reused across LPT_RSD_Batched objects (a new object is built per tracer per step)
    if n not in _POOL:
        _POOL[n] = ThreadPoolExecutor(max_workers=n)
    return _POOL[n]


class LPT_RSD_Batched(LPT_RSD):
    """Drop-in subclass of LPT_RSD with a vectorized make_pltable.
    Everything else (combine_bias_terms_*, setup_rsd_facs, ...) is inherited.
    """

    def __init__(self, *args, k_chunk=None, n_workers='auto', **kwargs):
        # k_chunk: evaluate p_integrals in blocks of this many k (None = all at once). All-at-once makes the
        # (nk, jn, ncol, N) intermediates memory-bound for large N; blocks of ~16-32 k keep them cache-sized.
        # n_workers: evaluate the k blocks of each mu concurrently on this many threads (numpy and pyfftw release
        # the GIL). 'auto' (default) = OMP_NUM_THREADS; 1 = serial blocks. With n_workers > 1 and k_chunk None, k is split
        # into n_workers blocks.
        # Each worker thread gets its own SphericalBesselTransform (they hold FFTW work buffers), run with FFTW
        # threads = 1. Results are identical to the serial path (blocks are independent; same arithmetic).
        self.k_chunk = k_chunk
        if n_workers == 'auto':
            n_workers = int(os.getenv("OMP_NUM_THREADS", "1"))
        self.n_workers = max(1, int(n_workers or 1))
        super().__init__(*args, **kwargs)
        # Cache of batched SphericalBesselTransform objects, keyed by nk.
        # Each one has ncol = nk * num_power_components so one sph.sph() call
        # does the FFT for every (k, component) at once.
        self._batched_sph = {}

    # ------------------------------------------------------------------ #
    def _get_batched_sph(self, nk):
        # one transform object per (block size, thread): they own FFTW input/output buffers, so are not shareable
        key = (nk, threading.get_ident())
        if key not in self._batched_sph:
            threads = 1 if self.n_workers > 1 else int(os.getenv("OMP_NUM_THREADS", "1"))
            self._batched_sph[key] = SphericalBesselTransform(
                self.qint, L=self.jn,
                ncol=nk * self.num_power_components,
                threads=threads,
            )
        return self._batched_sph[key]

    def _p_integrals_blocks(self, kts, nmax):
        """p_integrals_batched over kts, split into blocks (k_chunk / n_workers), optionally run concurrently."""
        nk = len(kts)
        if self.n_workers > 1:
            # one block per worker, but never more than 24 k per block (larger blocks get memory-bound at large N)
            size = self.k_chunk or min(int(np.ceil(nk / self.n_workers)), 24)
        else:
            size = self.k_chunk or nk
        blocks = [kts[c:c + size] for c in range(0, nk, size)]
        if self.n_workers > 1 and len(blocks) > 1:
            res = list(_get_pool(self.n_workers).map(lambda b: self.p_integrals_batched(b, nmax=nmax), blocks))
        else:
            res = [self.p_integrals_batched(b, nmax=nmax) for b in blocks]
        return np.concatenate(res, axis=0)

    @staticmethod
    def _shift_down_batched(mat, m):
        """Shift a (nk, jn, N) array along the jn (axis=1) by m, zero-padding."""
        out = np.zeros_like(mat)
        jn = mat.shape[1]
        if m < jn:
            out[:, m:] = mat[:, :jn - m]
        return out

    # ------------------------------------------------------------------ #
    def _compute_G_arrays_batched(self, kv, nmax):
        """Batched _compute_G_arrays. kv shape (nk,); returns 9 arrays of
        shape (nk, jn, N)."""
        jn  = self.jn
        kq  = kv[:, None] * self.qint                               # (nk, N)
        # kpow[i, j] = kv[i]**(2j)
        kpow = kv[:, None] ** (2 * np.arange(jn + nmax))            # (nk, jn+nmax)
        # kpow_pf[i, j, n] = kpow[i, j] * powerfacs[j, n]
        kpow_pf = kpow[:, :, None] * self.powerfacs[None, :, :]     # (nk, jn+nmax, N)

        def _band(b):
            # (jn, jn+nmax) contracted against (nk, jn+nmax, N) → (nk, jn, N).
            return np.einsum('ij,kjn->kin', b, kpow_pf)

        kqi = 1.0 / kq[:, None, :]                                   # (nk, 1, N)
        G0       = _band(self._G0_band)
        dGdA     = _band(self._dG0dA_band)     * kqi
        dGdC     = _band(self._dG0dC_band)     * kqi
        d2GdA2   = _band(self._d2G0dA2_band)   * kqi**2
        d2GdCdA  = _band(self._d2G0dCdA_band)  * kqi**2
        d2GdC2   = _band(self._d2G0dC2_band)   * kqi**2
        d3GdA3   = _band(self._d3G0dA3_band)   * kqi**3
        d3GdCdA2 = _band(self._d3G0dCdA2_band) * kqi**3
        d4GdA4   = _band(self._d4G0dA4_band)   * kqi**4
        return G0, dGdA, dGdC, d2GdA2, d2GdCdA, d2GdC2, d3GdA3, d3GdCdA2, d4GdA4

    # ------------------------------------------------------------------ #
    def p_integrals_batched(self, kv, nmax=8, old=False):
        """Compute p_integrals at all kv values in one vectorised pass.

        Returns an ndarray of shape (nk, num_power_components) whose row j is
        exactly what `p_integrals(kv[j])` would have returned.
        """
        kv = np.asarray(kv, dtype=float)
        nk = len(kv)

        Kfac = self.Kfac
        f    = self.f
        nu   = self.nu
        Anu  = self.Anu
        Bnu  = self.Bnu

        ksq  = kv**2
        K    = kv * Kfac
        Ksq  = K**2
        Knfac = nu * (1 + f)
        D2 = self.D**2

        # Per-q exponential and per-k constant prefactor.
        expon    = np.exp(-0.5 * Ksq[:, None] * D2 * (self.XYlin - self.sigma))  # (nk, N)
        suppress = np.exp(-0.5 * Ksq * D2 * self.sigma)                           # (nk,)

        # A, C are (nk, N): qint and c/s are scalars or (N,).
        A = kv[:, None] * self.qint * self.c
        C = kv[:, None] * self.qint * self.s

        # All G arrays shape (nk, jn, N).
        (G0s, dGdAs, dGdCs, d2GdA2s, d2GdCdAs, d2GdC2s,
         d3GdA3s, d3GdCdA2s, d4GdA4s) = self._compute_G_arrays_batched(kv, nmax)

        _shift = self._shift_down_batched
        G0_1   = _shift(G0s, 1);      G0_2 = _shift(G0s, 2)
        G0_3   = _shift(G0s, 3);      G0_4 = _shift(G0s, 4)
        dGdA_1 = _shift(dGdAs, 1);    dGdA_2 = _shift(dGdAs, 2)
        dGdA_3 = _shift(dGdAs, 3)
        dGdC_1 = _shift(dGdCs, 1);    dGdC_2 = _shift(dGdCs, 2)
        d2GdA2_1 = _shift(d2GdA2s, 1); d2GdA2_2 = _shift(d2GdA2s, 2)
        d2GdCdA_1 = _shift(d2GdCdAs, 1)
        d2GdC2_1  = _shift(d2GdC2s, 1)
        d3GdA3_1  = _shift(d3GdA3s, 1)

        # Broadcast (nk, N) A, C up to (nk, 1, N) for arithmetic with (nk, jn, N).
        A_b = A[:, None, :]; C_b = C[:, None, :]
        A2_b = A_b * A_b;    A3_b = A2_b * A_b;    A4_b = A2_b * A2_b

        G01s = -(dGdAs + 0.5 * A_b * G0_1)
        G02s = -(d2GdA2s + A_b * dGdA_1 + 0.5 * G0_1 + 0.25 * A2_b * G0_2)
        G03s = (d3GdA3s + 1.5 * A_b * d2GdA2_1 + 1.5 * dGdA_1
                + 0.75 * A2_b * dGdA_2 + 0.75 * A_b * G0_2 + (A3_b / 8.) * G0_3)
        G04s = (d4GdA4s + 2 * A_b * d3GdA3_1 + 3 * d2GdA2_1
                + 1.5 * A2_b * d2GdA2_2 + 3 * A_b * dGdA_2 + 0.75 * G0_2
                + 0.5 * A3_b * dGdA_3 + 0.75 * A2_b * G0_3
                + (A4_b / 16.) * G0_4)
        G10s = dGdCs + 0.5 * C_b * G0_1
        G11s = d2GdCdAs + 0.5 * C_b * dGdA_1 + 0.5 * A_b * dGdC_1 + 0.25 * A_b * C_b * G0_2
        G20s = -(d2GdC2s + C_b * dGdC_1 + 0.5 * G0_1 + 0.25 * C_b * C_b * G0_2)
        G12s = -(d3GdCdA2s + 0.5 * C_b * d2GdA2_1 + A_b * d2GdCdA_1 + 0.5 * dGdC_1
                 + 0.5 * A_b * C_b * dGdA_2 + 0.25 * A2_b * dGdC_2 + 0.25 * C_b * G0_2
                 + (A2_b * C_b / 8.) * G0_3)

        # (nk, jn, N) throughout.
        mu0     = G0s
        mu1     = G01s
        mu2     = G02s
        mu3     = G03s
        mu4     = G04s
        nq1     = Anu * G01s + Bnu * G10s
        mu_nq1  = Anu * G02s + Bnu * G11s
        nq2     = Anu**2 * G02s + 2 * Anu * Bnu * G11s + Bnu**(2 + 2 * old) * G20s
        mu2_nq1 = Anu * G03s + Bnu * G12s

        # bias_integrands_all shape (nk, jn, ncol, N)
        Xg = self.Xlin_gt; Yg = self.Ylin_gt
        ncol = self.num_power_components
        bias_integrands_all = np.zeros((nk, self.jn, ncol, self.N))

        # Broadcast per-k scalars to (nk, 1, 1) for multiplying (nk, jn, N) arrays.
        Ksq3 = Ksq[:, None, None]
        ksq3 = ksq[:, None, None]
        k3   = kv[:, None, None]
        K3   = K[:, None, None]

        bias_integrands_all[:, :, 0, :] = mu0 - 0.5 * Ksq3 * (Xg * mu0 + Yg * mu2)
        bias_integrands_all[:, :, 0, :] += (
            -0.5 * ksq3 * (
                2 * (Kfac**2 + 2 * f * (1 + f) * nu**2) * mu0 * self.X13
                + 2 * (Kfac**2 * mu2 + 2 * f * Kfac * nu * mu_nq1) * self.Y13
                + (Kfac**2 + 2 * f * (1 + f) * nu**2 + f**2 * nu**2) * mu0 * self.X22
                + (Kfac**2 * mu2 + 2 * f * Kfac * nu * mu_nq1 + f**2 * nu**2 * nq2) * self.Y22
            )
            + Ksq3**2 / 8. * (Xg**2 * mu0 + 2 * Xg * Yg * mu2 + Yg**2 * mu4)
        )
        bias_integrands_all[:, :, 0, :] += 0.5 * k3**3 * (
            2 * Kfac * (Kfac**2 + f * (1 + f) * nu**2) * mu1 * self.V1
            + Kfac**2 * (Kfac * mu1 + f * nu * nq1) * self.V3
            + Kfac**2 * (Kfac * mu3 + f * nu * mu2_nq1) * self.T
        )

        bias_integrands_all[:, :, 1, :] = (
            -2 * K3 * (self.Ulin + self.U3) * mu1
            - Ksq3 * (self.X10 * mu0 + self.Y10 * mu2)
            - 4 * f * k3 * nu * self.U3 * nq1
            - f * ksq3 * nu * (self.X10 * Knfac * mu0 + Kfac * self.Y10 * mu_nq1)
            - 2 * K3 * self.Ulin * (-0.5 * Ksq3 * (Xg * mu1 + Yg * mu3))
        )

        bias_integrands_all[:, :, 2, :] = (
            self.corlin * (mu0 - 0.5 * Ksq3 * (Xg * mu0 + Yg * mu2))
            - Ksq3 * self.Ulin**2 * mu2
            - (K3 * mu1 + f * k3 * nu * nq1) * self.U11
        )

        bias_integrands_all[:, :, 3, :] = (
            -Ksq3 * self.Ulin**2 * mu2 - k3 * (Kfac * mu1 + f * nu * nq1) * self.U20
        )
        bias_integrands_all[:, :, 4, :] = -2 * K3 * self.Ulin * self.corlin * mu1
        bias_integrands_all[:, :, 5, :] = 0.5 * self.corlin**2 * mu0

        if self.shear or self.third_order:
            bias_integrands_all[:, :, 6, :] = (
                -Ksq3 * (self.Xs2 * mu0 + self.Ys2 * mu2)
                - 2 * k3 * (Kfac * mu1 + f * nu * nq1) * self.Us2
            )
            bias_integrands_all[:, :, 7, :] = -2 * K3 * self.V * mu1
            bias_integrands_all[:, :, 8, :] = self.chi * mu0
            bias_integrands_all[:, :, 9, :] = self.zeta * mu0

        if self.third_order:
            bias_integrands_all[:, :, 10, :] = -2 * K3 * self.Ub3 * mu1
            bias_integrands_all[:, :, 11, :] = 2 * self.theta * mu0

        if self.use_Pzel:
            bias_integrands_all[:, :, -1, :] = mu0 - 0.5 * Ksq3 * (Xg * mu0 + Yg * mu2)
        else:
            bias_integrands_all[:, :, -1, :] = self.corlin * mu0

        # IR exponent * (-2/(kq))^l, broadcast over component axis.
        kq_fac = -2.0 / (kv[:, None] * self.qint)                    # (nk, N)
        factor_l = expon[:, None, :] * (
            kq_fac[:, None, :] ** np.arange(self.jn)[None, :, None]
        )                                                             # (nk, jn, N)
        bias_integrands_all *= factor_l[:, :, None, :]
        self._subtract_linear(bias_integrands_all, kv, K, Ksq)
        # l=0: subtract the value at q=qint[-1] (large-q limit), per k and component.
        bias_integrands_all[:, 0] -= bias_integrands_all[:, 0, :, -1:]
        self._high_pass(bias_integrands_all)

        # Batched FFTs. One sph.sph call per l covers all (k, component) pairs.
        sph_bulk = self._get_batched_sph(nk)
        ret = np.zeros((nk, ncol))

        for l in range(self.jn):
            input_l = bias_integrands_all[:, l].reshape(nk * ncol, self.N)
            ktemps, bias_ffts = sph_bulk.sph(l, input_l)
            _interp_k_many(kv, ktemps, bias_ffts.reshape(nk, ncol, -1), out=ret)

        self._add_linear(ret, kv)
        return 4 * np.pi * suppress[:, None] * ret

    # Linear-subtraction steps (same as in LPT_RSD.p_integrals; bias_integrands_all is
    # (nk, jn, ncol, N), ret is (nk, ncol)); no-ops unless linear_subtraction.
    def _subtract_linear(self, bias_integrands_all, kv, K, Ksq):
        if self.linear_subtraction:           # same as LPT_RSD.p_integrals, vectorised over kv
            b = bias_integrands_all
            Xf = self.Xlin + self.Xlin_gt; Yf = self.Ylin + self.Ylin_gt; cK = self.c
            Ksq_, K_ = Ksq[:, None], K[:, None]
            b[:, 0, 0, :] -= -0.5 * Ksq_ * (Xf + Yf / 3.)
            b[:, 2, 0, :] -= 0.5 * Ksq_ * Yf * (cK**2 - 1. / 3.)
            b[:, 1, 1, :] -= -2. * K_ * self.Ulin * cK
            b[:, 0, 2, :] -= self.corlin
            b[:, 0, -1, :] -= self.corlin

    def _high_pass(self, bias_integrands_all):
        if self.linear_subtraction:
            bias_integrands_all *= self._taper[None, None, None, :]

    def _add_linear(self, ret, kv):
        if self.linear_subtraction:
            plin_k = self._plin_exact(kv) / (4. * np.pi)
            kais = 1. + self.f * self.nu**2
            ret[:, 0] += kais**2 * plin_k; ret[:, 1] += 2. * kais * plin_k; ret[:, 2] += plin_k; ret[:, -1] += plin_k

    # ------------------------------------------------------------------ #
    def make_pltable(self, f, apar=1, aperp=1, ngauss=3, kv=None,
                     kmin=1e-2, kmax=0.25, nk=50, nmax=5):
        """Vectorized make_pltable: one batched p_integrals call per ν."""
        nus, ws = np.polynomial.legendre.leggauss(2 * ngauss)
        nus_calc = nus[0:ngauss]

        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0, 0, 1))(nus)
        L4 = np.polynomial.legendre.Legendre((0, 0, 0, 0, 1))(nus)

        if kv is None:
            kv = np.logspace(np.log10(kmin), np.log10(kmax), nk)
        else:
            nk = len(kv)

        self.pknutable    = np.zeros((len(nus), nk, self.num_power_components + 6))
        self.presknutable = np.zeros((len(nus), nk))

        # AP-distorted (k_true, nu_true) at every GL node — saved for downstream
        # custom mu-dependent counter-terms.
        self.ktrue_table = np.zeros((len(nus), nk))
        self.nu_true_arr = np.zeros(len(nus))

        vol_fac = apar * aperp**2

        for ii, nu in enumerate(nus_calc):
            fac     = np.sqrt(1 + nu**2 * ((aperp / apar)**2 - 1))
            k_apfac = fac / aperp
            nu_true = nu * aperp / apar / fac

            self.setup_rsd_facs(f, nu_true, nmax=nmax)

            ktrues = k_apfac * kv                       # (nk,)
            pterms = self._p_integrals_blocks(ktrues, nmax)   # (nk, ncol)

            self.ktrue_table[ii, :] = ktrues
            self.nu_true_arr[ii]    = nu_true

            # Bias-basis columns.
            self.pknutable[ii, :, :-7] = pterms[:, :-1]
            # Counterterms.
            kt2 = ktrues**2
            self.pknutable[ii, :, -7] = kt2 * pterms[:, -1]
            self.pknutable[ii, :, -6] = kt2 * nu_true**2 * pterms[:, -1]
            self.pknutable[ii, :, -5] = kt2 * nu_true**4 * pterms[:, -1]
            self.pknutable[ii, :, -4] = kt2 * nu_true**6 * pterms[:, -1]
            # Stochastic terms.
            self.pknutable[ii, :, -3] = 1
            self.pknutable[ii, :, -2] = kt2 * nu_true**2
            self.pknutable[ii, :, -1] = kt2**2 * nu_true**4
            # IR-resummed damped linear (for bispectrum).
            self.presknutable[ii, :] = pterms[:, -1]

        # Mirror reflected ν half.
        self.pknutable[ngauss:, :, :] = np.flip(self.pknutable[0:ngauss], axis=0)
        self.presknutable[ngauss:, :] = np.flip(self.presknutable[0:ngauss], axis=0)
        # ktrue is even in nu (k_apfac depends on nu^2); nu_true is odd in nu.
        self.ktrue_table[ngauss:, :] = np.flip(self.ktrue_table[0:ngauss, :], axis=0)
        self.nu_true_arr[ngauss:]    = -np.flip(self.nu_true_arr[0:ngauss], axis=0)

        self.kv       = kv
        self.p0ktable = 0.5 * np.sum((ws * L0)[:, None, None] * self.pknutable, axis=0) / vol_fac
        self.p2ktable = 2.5 * np.sum((ws * L2)[:, None, None] * self.pknutable, axis=0) / vol_fac
        self.p4ktable = 4.5 * np.sum((ws * L4)[:, None, None] * self.pknutable, axis=0) / vol_fac

        self.pres0ktable = 0.5 * np.sum((ws * L0)[:, None] * self.presknutable, axis=0) / vol_fac
        self.pres2ktable = 2.5 * np.sum((ws * L2)[:, None] * self.presknutable, axis=0) / vol_fac
        self.pres4ktable = 4.5 * np.sum((ws * L4)[:, None] * self.presknutable, axis=0) / vol_fac

        # GL grid + Legendre values + AP volume factor — saved so downstream
        # likelihood code can build custom mu-dependent counter-term multipoles.
        self.gl_nus  = nus
        self.gl_ws   = ws
        self.gl_L0   = L0
        self.gl_L2   = L2
        self.gl_L4   = L4
        self.vol_fac = vol_fac

        # Free the per-thread transform objects (FFTW buffers, ~15 MB each x block sizes x threads): they are only needed
        # inside make_pltable, and keeping them on every LPT object (6 per step, several retained by caches) exhausted
        # node memory in production (OOM, w80_200 jobs 14028486-92, 2026-11-21).
        self._batched_sph = {}

        return 0
