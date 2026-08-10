"""Deterministic property tests for torch-free inference primitives."""

from __future__ import annotations

import numpy as np
import pytest

from fastrho.filtering import basic_filtering
from fastrho.stitching import combine_gaussian_moments, positive_hann_weights


@pytest.mark.parametrize("length", range(1, 17))
def test_positive_hann_weights_preserve_every_token_and_symmetry(length: int) -> None:
    weights = positive_hann_weights(length)
    assert weights.shape == (length,)
    assert weights.dtype.kind == "f"
    assert np.isfinite(weights).all()
    assert np.all(weights > 0)
    assert np.all(weights <= 1)
    assert np.allclose(weights, weights[::-1], rtol=0, atol=1e-15)


@pytest.mark.parametrize("seed", range(8))
def test_gaussian_stitching_obeys_total_variance_identity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n_components = int(rng.integers(2, 9))
    width = int(rng.integers(1, 50))
    weights = rng.uniform(0.01, 3.0, size=(n_components, width))
    means = rng.normal(size=(n_components, width))
    variances = rng.lognormal(mean=-2.0, sigma=1.0, size=(n_components, width))

    weight_sum = weights.sum(axis=0)
    first = np.sum(weights * means, axis=0)
    second = np.sum(weights * (variances + means**2), axis=0)
    observed_mean, observed_variance = combine_gaussian_moments(first, second, weight_sum)

    expected_mean = np.average(means, axis=0, weights=weights)
    expected_variance = np.average(variances + means**2, axis=0, weights=weights) - expected_mean**2
    assert np.allclose(observed_mean, expected_mean, rtol=1e-13, atol=1e-13)
    assert np.allclose(observed_variance, expected_variance, rtol=1e-13, atol=1e-13)
    assert np.all(observed_variance > 0)


@pytest.mark.parametrize("seed", range(8))
def test_basic_filtering_retains_exactly_biallelic_segregating_sites(seed: int) -> None:
    rng = np.random.default_rng(seed)
    genotypes = rng.integers(0, 2, size=(12, 20), dtype=np.int8)
    genotypes[:, 0] = 0
    genotypes[:, 1] = 1
    genotypes[0, 2] = 2
    positions = np.cumsum(rng.uniform(1, 1000, size=20))
    original_genotypes = genotypes.copy()
    original_positions = positions.copy()

    observed_genotypes, observed_positions = basic_filtering(genotypes, positions)
    non_biallelic = np.any(original_genotypes > 1, axis=0)
    allele_count = original_genotypes.sum(axis=0)
    fixed = (allele_count == 0) | (allele_count == original_genotypes.shape[0])
    expected = ~(non_biallelic | fixed)

    assert np.array_equal(observed_genotypes, original_genotypes[:, expected])
    assert np.array_equal(observed_positions, original_positions[expected])
    assert observed_genotypes.dtype == np.int8
    assert np.array_equal(genotypes, original_genotypes), "filter mutated caller genotype data"
    assert np.array_equal(positions, original_positions), "filter mutated caller positions"
