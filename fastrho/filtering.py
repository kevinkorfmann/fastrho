"""Shared, explicit genotype filtering."""

from __future__ import annotations

import numpy as np


def basic_filtering(
    gm: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return biallelic, segregating sites from a haplotype matrix.

    ``gm`` must be shaped ``(n_haplotypes, n_sites)`` and contain only allele
    codes 0/1. Missing or multiallelic values are rejected rather than silently
    converted to reference alleles.
    """
    gm = np.asarray(gm)
    positions = np.asarray(positions, dtype=np.float64)
    if gm.ndim != 2:
        raise ValueError("genotype matrix must have shape (n_haplotypes, n_sites)")
    if positions.ndim != 1 or positions.size != gm.shape[1]:
        raise ValueError("positions must be one-dimensional and match genotype sites")
    if np.any(gm < 0):
        raise ValueError("missing genotypes must be removed or explicitly handled before filtering")
    non_biallelic = np.any(gm > 1, axis=0)
    allele_count = gm.sum(axis=0)
    fixed = (allele_count == 0) | (allele_count == gm.shape[0])
    keep = ~(non_biallelic | fixed)
    return gm[:, keep].astype(np.int8, copy=False), positions[keep]
