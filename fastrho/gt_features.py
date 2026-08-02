"""Phase-invariant SNP-token features for fastrho (unphased / genotype mode).

Identical 17-dim token layout to features.SNPTokenFeaturizer, but every LD quantity is
computed from diploid genotype *dosages* (0/1/2) instead of haplotypes, using composite
LD (Rogers & Huff 2009). Under random mating the squared dosage correlation estimates the
haplotype r^2, and the composite covariance D = cov(g_i, g_j)/2 estimates the gametic D_AB,
so the reconstructed two-locus config fractions match the haplotype ones in expectation.

Because dosage = sum of the two alleles, these features do not depend on phase: the output
is bit-identical on phased data and on the same data with phase scrambled. A model trained
on these features therefore has no phased/unphased gap.
"""

from __future__ import annotations

import numpy as np

from fastrho.features import (
    FeatureConfig,
    _assign_band,
    _rolling_mean,
    n_features,
    njit,
)


@njit(cache=True, fastmath=True)
def _snp_gt_ld_features(D, pos, md, vd, pA, radii, max_neigh,
                        disjoint=False, stride_after=0):
    """D: (S, m) float dosages 0/1/2, SNP-major; md (S,) mean dosage; vd (S,) dosage var;
    pA (S,) derived freq. Composite-LD analogue of features._snp_ld_features.
    `disjoint` / `stride_after` behave as in features._snp_ld_features (long-range LD).

    Returns mean_r2 (S,R), npairs (S,R), r2_next (S,), cfg (S,4)=[AB,Ab,aB,ab].
    """
    S, m = D.shape
    R = len(radii)
    rmax = radii[R - 1]
    ld_sum = np.zeros((S, R))
    ld_cnt = np.zeros((S, R))
    r2_next = np.zeros(S)
    cfg = np.zeros((S, 4))
    for i in range(S):
        ai = pA[i]
        if ai <= 0.0 or ai >= 1.0 or vd[i] <= 0.0:
            continue
        # forward
        j = i + 1
        seen = 0
        step = 1
        acc = 0
        while j < S and (pos[j] - pos[i]) < rmax and seen < max_neigh:
            seen += 1
            aj = pA[j]
            if 0.0 < aj < 1.0 and vd[j] > 0.0:
                dot = 0.0
                for a in range(m):
                    dot += D[i, a] * D[j, a]
                cov = dot / m - md[i] * md[j]              # genotypic covariance
                r2 = (cov * cov) / (vd[i] * vd[j])         # = corr(dosage)^2
                _assign_band(ld_sum, ld_cnt, i, pos[j] - pos[i], r2, radii, R, disjoint)
                if j == i + 1:
                    r2_next[i] = r2
                    Dab = 0.5 * cov                        # composite -> gametic D_AB
                    pAB = ai * aj + Dab
                    cfg[i, 0] = pAB
                    cfg[i, 1] = ai - pAB
                    cfg[i, 2] = aj - pAB
                    cfg[i, 3] = 1.0 - ai - aj + pAB
                acc += 1
                if stride_after > 0 and acc % stride_after == 0 and step < 4096:
                    step *= 2
            j += step
        # backward
        j = i - 1
        seen = 0
        step = 1
        acc = 0
        while j >= 0 and (pos[i] - pos[j]) < rmax and seen < max_neigh:
            seen += 1
            aj = pA[j]
            if 0.0 < aj < 1.0 and vd[j] > 0.0:
                dot = 0.0
                for a in range(m):
                    dot += D[i, a] * D[j, a]
                cov = dot / m - md[i] * md[j]
                r2 = (cov * cov) / (vd[i] * vd[j])
                _assign_band(ld_sum, ld_cnt, i, pos[i] - pos[j], r2, radii, R, disjoint)
                acc += 1
                if stride_after > 0 and acc % stride_after == 0 and step < 4096:
                    step *= 2
            j -= step
    mean_r2 = ld_sum / np.maximum(ld_cnt, 1.0)
    return mean_r2, ld_cnt, r2_next, cfg


class GTTokenFeaturizer:
    """Phase-invariant featurizer: (gm, positions, meta) -> dict(tokens, positions).

    gm is the haplotype matrix (n_hap, S); consecutive haplotypes are paired into diploid
    dosages. On already-unphased data, pass the scrambled gm -- the dosages (and hence the
    tokens) are unchanged.

    fold=True additionally makes the tokens polarization-invariant (ancestral/derived state
    unknown): every SNP is relabelled to the minor-allele convention (dosage D -> 2-D where
    the counted allele is the major one), so derived_af becomes the folded MAF and the
    two-locus configs use a canonical labelling. r^2, npairs, theta_pi and distances are
    already polarization-invariant. The result is invariant to BOTH phase and polarization.
    """

    def __init__(self, config: FeatureConfig = FeatureConfig(), fold: bool = False):
        self.cfg = config
        self.fold = fold

    @property
    def n_features(self) -> int:
        return n_features(self.cfg)

    def __call__(self, gm: np.ndarray, positions: np.ndarray, meta: dict) -> dict:
        cfg = self.cfg
        gm = np.asarray(gm)
        positions = np.asarray(positions)
        if gm.ndim != 2 or positions.ndim != 1 or gm.shape[1] != positions.size:
            raise ValueError("gm must be (n_haplotypes, n_sites) and match positions")
        n_hap = gm.shape[0]
        S = gm.shape[1]
        if n_hap < 2:
            raise ValueError("at least one diploid individual is required")
        pos = positions.astype(np.float64)
        if S > 1 and np.any(np.diff(pos) <= 0):
            raise ValueError("positions must be strictly increasing within one contig")
        if S == 0:
            return {"tokens": np.zeros((0, self.n_features), np.float32),
                    "positions": pos}

        # pair haplotypes -> diploid dosages (drop an odd trailing haplotype)
        m = n_hap // 2
        nn = 2 * m
        D = (gm[0:2 * m:2].astype(np.float64) + gm[1:2 * m:2].astype(np.float64))  # (m, S)
        pA = D.sum(0) / (2.0 * m)                       # derived allele freq
        if self.fold:
            # polarization unknown: count the minor allele at every SNP (D -> 2-D if >0.5)
            flip = pA > 0.5
            D[:, flip] = 2.0 - D[:, flip]
            pA = D.sum(0) / (2.0 * m)

        dnext = np.empty(S)
        dnext[:-1] = np.diff(pos)
        dnext[-1] = 0.0
        dprev = np.empty(S)
        dprev[1:] = np.diff(pos)
        dprev[0] = 0.0

        Dt = np.ascontiguousarray(D.T.astype(np.float64))     # (S, m) SNP-major
        md = Dt.mean(1)
        vd = Dt.var(1)
        radii = np.asarray(cfg.ld_radii, dtype=np.float64)
        mean_r2, npairs, r2_next, configs = _snp_gt_ld_features(
            Dt, pos, md, vd, pA, radii, cfg.max_neighbors,
            cfg.disjoint_bands, cfg.stride_after)
        if cfg.r2_debias:
            floor = 1.0 / nn
            mean_r2 = np.maximum(mean_r2 - floor, 0.0)
            r2_next = np.maximum(r2_next - floor, 0.0)

        # local diversity: theta_pi from allele freq (phase-free), and unique-genotype
        # fraction as the phase-invariant analogue of unique-haplotype fraction.
        pi_site = 2.0 * (pA * nn) * (nn - pA * nn) / (nn * (nn - 1))
        k = cfg.neigh_snps
        local_pi = _rolling_mean(pi_site, k)
        local_ngeno = self._local_n_geno_frac(D.astype(np.int16), k, nn)

        maf = np.minimum(pA, 1.0 - pA)
        parts = [
            np.log10(dnext + 1.0)[:, None],
            np.log10(dprev + 1.0)[:, None],
            pA[:, None],
            maf[:, None],
            mean_r2,
            np.log1p(npairs),
            r2_next[:, None],
            configs,
            local_pi[:, None],
            local_ngeno[:, None],
        ]
        if cfg.sfs_shape:
            rare = (maf <= (2.0 / nn)).astype(np.float64)
            parts += [_rolling_mean(rare, k)[:, None]]
        tokens = np.concatenate(parts, axis=1).astype(np.float32)
        return {"tokens": tokens, "positions": pos}

    @staticmethod
    def _local_n_geno_frac(D: np.ndarray, k: int, nn: int) -> np.ndarray:
        # D: (m, S) dosages; unique diploid genotype rows in +/-k window, scaled by n_hap
        S = D.shape[1]
        out = np.empty(S)
        for i in range(S):
            lo = max(0, i - k)
            hi = min(S, i + k + 1)
            out[i] = len(np.unique(D[:, lo:hi], axis=0)) / nn
        return out
