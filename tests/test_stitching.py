import numpy as np

from fastrho.stitching import combine_gaussian_moments, positive_hann_weights


def test_chunk_weights_never_drop_endpoints():
    for length in (1, 2, 3, 32, 1024):
        weights = positive_hann_weights(length)
        assert weights.shape == (length,)
        assert np.all(weights > 0)


def test_gaussian_stitching_includes_between_chunk_disagreement():
    # Two equally weighted N(-1, 1) and N(+1, 1) predictions form a mixture
    # with mean 0 and variance E[var + mu^2] = 2, not variance 1.
    mean_sum = np.array([0.0])
    second_sum = np.array([(1.0 + 1.0) + (1.0 + 1.0)])
    weight_sum = np.array([2.0])
    mean, variance = combine_gaussian_moments(mean_sum, second_sum, weight_sum)
    assert np.allclose(mean, 0.0)
    assert np.allclose(variance, 2.0)
