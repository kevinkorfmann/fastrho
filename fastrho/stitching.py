"""Torch-free helpers for stitching overlapping Gaussian chunk predictions."""

from __future__ import annotations

import numpy as np


def positive_hann_weights(length: int) -> np.ndarray:
    """Hann interior weights with strictly positive endpoints."""
    if length < 1:
        raise ValueError("chunk length must be positive")
    return np.hanning(length + 2)[1:-1]


def combine_gaussian_moments(
    weighted_mean: np.ndarray,
    weighted_second_moment: np.ndarray,
    weight_sum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover mixture mean/variance from accumulated first and second moments."""
    if np.any(np.asarray(weight_sum) <= 0):
        raise ValueError("every output token must receive positive chunk weight")
    mean = np.asarray(weighted_mean) / weight_sum
    second = np.asarray(weighted_second_moment) / weight_sum
    variance = np.maximum(second - mean * mean, 1e-12)
    return mean, variance
