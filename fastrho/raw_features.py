"""Raw-genotype per-SNP features for the ReLERNN-seq2seq steelman baseline.

Mirrors ReLERNN's input philosophy: each SNP token is the raw genotype vector across
haplotypes (encoded -1 ancestral / +1 derived / 0 pad, ReLERNN's convention), padded to a
fixed number of haplotypes, plus minimal positional features. No LD summaries -- that is the
point: the GRU must learn structure from the raw matrix, exactly as ReLERNN does. Haplotypes
are given a consistent lexicographic order per region (ReLERNN's optional sortInds intent).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RawFeatureConfig:
    max_hap: int = 200          # pad/truncate haplotype axis to this many


def n_features(cfg: RawFeatureConfig = RawFeatureConfig()) -> int:
    return cfg.max_hap + 3      # genotype vector + log_dnext + log_dprev + derived_af


def feature_names(cfg: RawFeatureConfig = RawFeatureConfig()) -> list[str]:
    return [f"hap_{i}" for i in range(cfg.max_hap)] + ["log_dpos_next", "log_dpos_prev", "derived_af"]


class RawGenotypeFeaturizer:
    """Callable: (gm, positions, meta) -> dict(tokens (S, max_hap+3), positions)."""

    def __init__(self, config: RawFeatureConfig = RawFeatureConfig()):
        self.cfg = config

    @property
    def n_features(self) -> int:
        return n_features(self.cfg)

    def __call__(self, gm: np.ndarray, positions: np.ndarray, meta: dict) -> dict:
        H = self.cfg.max_hap
        n, S = gm.shape
        pos = positions.astype(np.float64)
        if S == 0:
            return {"tokens": np.zeros((0, self.n_features), np.float32), "positions": pos}

        # consistent haplotype ordering (group similar haplotypes) for a stable matrix
        order = np.lexsort(gm.T)                      # permutation of the n haplotypes
        g = (2.0 * gm[order].astype(np.float32) - 1.0)  # (n, S) in {-1, +1}

        geno = np.zeros((S, H), np.float32)
        m = min(n, H)
        geno[:, :m] = g[:m].T                         # (S, max_hap), zero-padded

        dnext = np.empty(S)
        dnext[:-1] = np.diff(pos)
        dnext[-1] = 0.0
        dprev = np.empty(S)
        dprev[1:] = np.diff(pos)
        dprev[0] = 0.0
        af = gm.sum(0).astype(np.float64) / n

        tokens = np.concatenate([
            geno,
            np.log10(dnext + 1.0)[:, None],
            np.log10(dprev + 1.0)[:, None],
            af[:, None],
        ], axis=1).astype(np.float32)
        return {"tokens": tokens, "positions": pos}
