"""Per-SNP-token features for fastrho.

We treat SNPs as tokens (a "language model over the genome"): a bidirectional Mamba
reads the SNP sequence and predicts a recombination rate per SNP-*interval*. This is the
natural granularity for recombination — the signal lives *between* adjacent SNPs (it is
exactly pyrho's two-locus sufficient statistic) — and it lets the SSM handle a variable
number of real SNPs natively instead of binning into arbitrary bp windows.

Each token (SNP i) carries permutation-invariant, sample-size-robust features describing
the interval to its right, dominated by linkage information:

  * log distance to the next / previous SNP (bp)         -- rate scales with bp distance
  * derived allele frequency and folded MAF              -- polarization / diversity
  * two-locus haplotype config fractions for (i, i+1)    -- pyrho's exact LD statistic
  * multi-lag r^2 to SNP i+1..i+L (and i-1)              -- the LD-decay signal
  * local theta_pi and haplotype diversity (+/- k SNPs)  -- Ne signal for rho<->r

The conv/SSM stem (model side) learns higher-order structure on top of these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return deco


@dataclass(frozen=True)
class FeatureConfig:
    # bp radii for neighbourhood-aggregated LD (the robust recombination signal:
    # single-pair r^2 is too noisy; mean r^2 over a neighbourhood tracks the rate).
    ld_radii: tuple[int, ...] = (5000, 25000, 50000)
    neigh_snps: int = 8        # +/- k SNPs for local diversity stats
    max_neighbors: int = 200   # cap LD pairs per direction (keeps dense regions fast)
    # --- long-range LD options (dog / severe-bottleneck regime) ---------------
    # disjoint_bands: assign each pair to a SINGLE band [radii[k-1], radii[k]) instead
    #   of every enclosing radius. Makes the widest band a clean long-range-LD statistic
    #   (cumulative radii are swamped by near pairs, which hides Mb-scale block LD).
    # stride_after: once this many polymorphic neighbours have been compared in a
    #   direction, geometrically double the index step. A bounded `max_neighbors`
    #   budget then SPANS up to radii[-1] (e.g. 1-5 Mb breed LD) instead of stalling
    #   in the first few kb. 0 => contiguous walk (original behaviour).
    # Defaults below reproduce the original cumulative/contiguous behaviour exactly.
    disjoint_bands: bool = False
    stride_after: int = 0
    # rich_ld: append per-band two-locus statistics that KEEP sharpening with sample size
    # (mean r^2 and config fractions are deliberately n-robust and saturate at large n,
    # which is exactly the information pyrho's exact two-locus likelihood keeps exploiting).
    #   * frac_4gamete[band]: fraction of pairs showing all four haplotypes (the classic
    #       Hudson-Kaplan recombination signal; rises toward truth as n grows)
    #   * mean_|D'|[band]:   normalized LD that, unlike r^2, is not deflated by unequal
    #       marginal frequencies -- a complementary LD-decay signal
    # Adds 2*len(ld_radii) features. Off by default so the base 17-feature path is unchanged.
    rich_ld: bool = False
    # Domain-randomized feature options. Both default off so legacy 17-feature
    # checkpoints retain their exact layout.
    r2_debias: bool = False
    sfs_shape: bool = False


def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centered rolling mean over +/- k elements; length == len(x) for any size."""
    S = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(S)
    lo = np.maximum(idx - k, 0)
    hi = np.minimum(idx + k + 1, S)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def feature_names(cfg: FeatureConfig = FeatureConfig()) -> list[str]:
    names = ["log_dpos_next", "log_dpos_prev", "derived_af", "maf"]
    names += [f"mean_r2_{r}" for r in cfg.ld_radii]
    names += [f"log_npairs_{r}" for r in cfg.ld_radii]
    names += ["r2_fwd_1"]
    names += ["cfg_AB", "cfg_Ab", "cfg_aB", "cfg_ab"]
    names += ["local_theta_pi", "local_n_hap_frac"]
    if cfg.sfs_shape:
        names += ["local_rare_frac"]
    if cfg.rich_ld:
        names += [f"frac_4gam_{r}" for r in cfg.ld_radii]
        names += [f"mean_dprime_{r}" for r in cfg.ld_radii]
    return names


def n_features(cfg: FeatureConfig = FeatureConfig()) -> int:
    return len(feature_names(cfg))


def mean_r2_slice(cfg: FeatureConfig = FeatureConfig()) -> slice:
    """Column slice of the per-radius ``mean_r2`` features in the token layout.

    Derived by NAME from :func:`feature_names`, so it stays correct if features are inserted
    before/after the LD bands or the fold path reorders columns. Prefer this over a hard-coded
    ``tokens[:, 4:4+len(ld_radii)]`` when reading the LD-decay bands (e.g. fig_dog panel a)."""
    names = feature_names(cfg)
    idx = [i for i, nm in enumerate(names) if nm.startswith("mean_r2_")]
    if not idx:
        raise ValueError("no mean_r2 features in this config")
    return slice(idx[0], idx[-1] + 1)


# ---------------------------------------------------------------------------
# LD / two-locus kernel (numba)
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _assign_band(ld_sum, ld_cnt, i, d, r2, radii, R, disjoint):
    """Accumulate r2 into band(s) for distance d. Cumulative (every enclosing radius)
    unless `disjoint`, in which case only the first matching band [radii[k-1],radii[k])."""
    for ri in range(R):
        if d < radii[ri]:
            ld_sum[i, ri] += r2
            ld_cnt[i, ri] += 1.0
            if disjoint:
                return


@njit(cache=True, fastmath=True)
def _assign_band_rich(fg_sum, dp_sum, i, d, fourgam, dprime, radii, R, disjoint):
    """Accumulate the 4-gamete indicator and |D'| into band(s) for distance d, using the
    same cumulative/disjoint rule as _assign_band (shares ld_cnt for the denominator)."""
    for ri in range(R):
        if d < radii[ri]:
            fg_sum[i, ri] += fourgam
            dp_sum[i, ri] += dprime
            if disjoint:
                return


@njit(cache=True, fastmath=True)
def _snp_ld_features(g, pos, pA, radii, max_neigh, disjoint=False, stride_after=0):
    """g: (S, n) float32 SNP-major 0/1; pos (S,) bp; pA (S,) derived freqs.

    For each SNP i, aggregate r^2 over neighbour pairs within each bp radius (both
    directions, at most max_neigh polymorphic comparisons per direction). With
    stride_after>0 the index step doubles every `stride_after` comparisons so a bounded
    budget reaches radii[-1] (long-range/bottleneck LD); with disjoint=True each pair
    lands in a single distance band. Returns:
      mean_r2 (S, R), npairs (S, R), r2_next (S,), cfg (S, 4)=[AB, Ab, aB, ab] for (i,i+1),
      mean_fourgam (S, R), mean_dprime (S, R).
    """
    S, n = g.shape
    R = len(radii)
    rmax = radii[R - 1]
    ld_sum = np.zeros((S, R))
    ld_cnt = np.zeros((S, R))
    fg_sum = np.zeros((S, R))
    dp_sum = np.zeros((S, R))
    r2_next = np.zeros(S)
    cfg = np.zeros((S, 4))
    for i in range(S):
        ai = pA[i]
        if ai <= 0.0 or ai >= 1.0:
            continue
        ci = ai * n
        # forward neighbours
        j = i + 1
        seen = 0
        step = 1
        acc = 0
        while j < S and (pos[j] - pos[i]) < rmax and seen < max_neigh:
            seen += 1
            aj = pA[j]
            if 0.0 < aj < 1.0:
                cAB = 0.0
                for a in range(n):
                    cAB += g[i, a] * g[j, a]
                pab = cAB / n
                num = pab - ai * aj
                den = ai * (1.0 - ai) * aj * (1.0 - aj)
                if den > 0.0:
                    r2 = num * num / den
                    _assign_band(ld_sum, ld_cnt, i, pos[j] - pos[i], r2, radii, R, disjoint)
                    # 4-gamete test on integer haplotype counts
                    cj = aj * n
                    fourgam = 1.0 if (cAB > 0.5 and (ci - cAB) > 0.5 and
                                      (cj - cAB) > 0.5 and (n - ci - cj + cAB) > 0.5) else 0.0
                    if num >= 0.0:
                        dmax = ai * (1.0 - aj)
                        o = (1.0 - ai) * aj
                        if o < dmax:
                            dmax = o
                    else:
                        dmax = ai * aj
                        o = (1.0 - ai) * (1.0 - aj)
                        if o < dmax:
                            dmax = o
                    dprime = (num / dmax if num >= 0.0 else -num / dmax) if dmax > 1e-12 else 0.0
                    _assign_band_rich(fg_sum, dp_sum, i, pos[j] - pos[i], fourgam, dprime,
                                      radii, R, disjoint)
                    if j == i + 1:
                        r2_next[i] = r2
                        cfg[i, 0] = pab
                        cfg[i, 1] = ai - pab
                        cfg[i, 2] = aj - pab
                        cfg[i, 3] = 1.0 - ai - aj + pab
                    acc += 1
                    if stride_after > 0 and acc % stride_after == 0 and step < 4096:
                        step *= 2
            j += step
        # backward neighbours
        j = i - 1
        seen = 0
        step = 1
        acc = 0
        while j >= 0 and (pos[i] - pos[j]) < rmax and seen < max_neigh:
            seen += 1
            aj = pA[j]
            if 0.0 < aj < 1.0:
                cAB = 0.0
                for a in range(n):
                    cAB += g[i, a] * g[j, a]
                pab = cAB / n
                num = pab - ai * aj
                den = ai * (1.0 - ai) * aj * (1.0 - aj)
                if den > 0.0:
                    r2 = num * num / den
                    _assign_band(ld_sum, ld_cnt, i, pos[i] - pos[j], r2, radii, R, disjoint)
                    cj = aj * n
                    fourgam = 1.0 if (cAB > 0.5 and (ci - cAB) > 0.5 and
                                      (cj - cAB) > 0.5 and (n - ci - cj + cAB) > 0.5) else 0.0
                    if num >= 0.0:
                        dmax = ai * (1.0 - aj)
                        o = (1.0 - ai) * aj
                        if o < dmax:
                            dmax = o
                    else:
                        dmax = ai * aj
                        o = (1.0 - ai) * (1.0 - aj)
                        if o < dmax:
                            dmax = o
                    dprime = (num / dmax if num >= 0.0 else -num / dmax) if dmax > 1e-12 else 0.0
                    _assign_band_rich(fg_sum, dp_sum, i, pos[i] - pos[j], fourgam, dprime,
                                      radii, R, disjoint)
                    acc += 1
                    if stride_after > 0 and acc % stride_after == 0 and step < 4096:
                        step *= 2
            j -= step
    denom = np.maximum(ld_cnt, 1.0)
    mean_r2 = ld_sum / denom
    mean_fourgam = fg_sum / denom
    mean_dprime = dp_sum / denom
    return mean_r2, ld_cnt, r2_next, cfg, mean_fourgam, mean_dprime


# ---------------------------------------------------------------------------
# Featurizer
# ---------------------------------------------------------------------------

class SNPTokenFeaturizer:
    """Callable: (gm, positions, meta) -> dict(tokens, positions)."""

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.cfg = config

    @property
    def n_features(self) -> int:
        return n_features(self.cfg)

    def __call__(self, gm: np.ndarray, positions: np.ndarray, meta: dict) -> dict:
        cfg = self.cfg
        gm = np.asarray(gm)
        positions = np.asarray(positions)
        if gm.ndim != 2 or positions.ndim != 1 or gm.shape[1] != positions.size:
            raise ValueError("gm must be (n_haplotypes, n_sites) and match positions")
        n = gm.shape[0]
        S = gm.shape[1]
        if n < 2:
            raise ValueError("at least two haplotypes are required")
        pos = positions.astype(np.float64)
        if S > 1 and np.any(np.diff(pos) <= 0):
            raise ValueError("positions must be strictly increasing within one contig")

        if S == 0:
            return {"tokens": np.zeros((0, self.n_features), np.float32),
                    "positions": pos.astype(np.float64)}

        pA = gm.sum(0).astype(np.float64) / n

        # distances to neighbours (bp)
        dnext = np.empty(S)
        dnext[:-1] = np.diff(pos)
        dnext[-1] = 0.0
        dprev = np.empty(S)
        dprev[1:] = np.diff(pos)
        dprev[0] = 0.0

        g = np.ascontiguousarray(gm.T.astype(np.float32))   # (S, n)
        radii = np.asarray(cfg.ld_radii, dtype=np.float64)
        mean_r2, npairs, r2_next, configs, mean_fourgam, mean_dprime = _snp_ld_features(
            g, pos, pA, radii, cfg.max_neighbors,
            cfg.disjoint_bands, cfg.stride_after)
        if cfg.r2_debias:
            floor = 1.0 / n
            mean_r2 = np.maximum(mean_r2 - floor, 0.0)
            r2_next = np.maximum(r2_next - floor, 0.0)

        # local diversity in +/- k SNP neighbourhood (robust for any S)
        pi_site = 2.0 * (pA * n) * (n - pA * n) / (n * (n - 1))   # per-site pi
        k = cfg.neigh_snps
        local_pi = _rolling_mean(pi_site, k)
        local_nhap = self._local_n_hap_frac(gm, k, n)

        maf = np.minimum(pA, 1.0 - pA)
        parts = [
            np.log10(dnext + 1.0)[:, None],
            np.log10(dprev + 1.0)[:, None],
            pA[:, None],
            maf[:, None],
            mean_r2,                              # (S, R)
            np.log1p(npairs),                     # (S, R)
            r2_next[:, None],
            configs,                              # (S, 4)
            local_pi[:, None],
            local_nhap[:, None],
        ]
        if cfg.sfs_shape:
            rare = (maf <= (2.0 / n)).astype(np.float64)
            parts += [_rolling_mean(rare, k)[:, None]]
        if cfg.rich_ld:
            parts += [mean_fourgam, mean_dprime]  # (S, R) each -- n-sensitive two-locus stats
        tokens = np.concatenate(parts, axis=1).astype(np.float32)
        return {"tokens": tokens, "positions": pos}

    @staticmethod
    def _local_n_hap_frac(gm: np.ndarray, k: int, n: int) -> np.ndarray:
        S = gm.shape[1]
        out = np.empty(S)
        for i in range(S):
            lo = max(0, i - k)
            hi = min(S, i + k + 1)
            sub = gm[:, lo:hi]
            out[i] = len(np.unique(sub, axis=0)) / n
        return out
