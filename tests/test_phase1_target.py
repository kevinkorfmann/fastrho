"""Phase-1 validation: exact per-window recombination target + map generation.

These tests use only numpy + msprime (no torch/mamba), so they run anywhere.
"""

import numpy as np
import pytest

from fastrho.preprocess import windowed_recombination_rate
from fastrho.simulate import (
    RecombPriors,
    _dump_one,
    make_recombination_map,
    simulate_region,
    simulate_stdpopsim_region,
)


def test_constant_map_target_is_flat():
    L, w = 100_000, 1000
    rate = 1.5e-8
    y = windowed_recombination_rate([0.0, L], [rate], w, L)
    assert y.shape == (L // w,)
    assert np.allclose(y, rate)


def test_two_segment_window_average():
    # map: 0..1000 @ 1e-8, 1000..2000 @ 3e-8; window 0..2000 -> mean 2e-8
    L, w = 2000, 2000
    y = windowed_recombination_rate([0.0, 1000.0, 2000.0], [1e-8, 3e-8], w, L)
    assert np.allclose(y, [2e-8])


def test_partial_segment_weighting():
    # window 0..1000 over map 0..400@1e-8, 400..1000@2e-8
    # mean = (400*1e-8 + 600*2e-8)/1000 = 1.6e-8
    y = windowed_recombination_rate([0.0, 400.0, 1000.0], [1e-8, 2e-8], 1000, 1000)
    assert np.allclose(y, [1.6e-8])


def test_target_matches_ratemap_mean_average():
    # cross-check against msprime.RateMap.mean_rate over the whole region
    rng = np.random.default_rng(0)
    L = 200_000
    rm = make_recombination_map(L, rng, kind="gp", mean_rate=1e-8)
    y = windowed_recombination_rate(rm.position, rm.rate, 1000, L)
    # span-weighted mean of window means == overall mean rate of the map
    overall = np.average(rm.rate, weights=np.diff(rm.position))
    assert np.isclose(y.mean(), overall, rtol=1e-6)


def test_map_kinds_are_valid_ratemaps():
    rng = np.random.default_rng(1)
    for kind in ("constant", "gp", "hotspot"):
        rm = make_recombination_map(1_000_000, rng, kind=kind, mean_rate=1e-8)
        assert rm.position[0] == 0.0
        assert rm.position[-1] == 1_000_000.0
        assert np.all(np.diff(rm.position) > 0)
        assert np.all(np.isfinite(rm.rate))
        assert np.all(rm.rate > 0)


def test_hotspot_has_elevated_windows():
    rng = np.random.default_rng(3)
    L = 1_000_000
    rm = make_recombination_map(L, rng, kind="hotspot", mean_rate=1e-8)
    y = windowed_recombination_rate(rm.position, rm.rate, 2000, L)
    assert y.max() > 5 * np.median(y)   # at least one clear hotspot window


def test_simulate_region_end_to_end():
    priors = RecombPriors(sequence_length=200_000, window_size=2000)
    ts, rate_map, meta = simulate_region(42, priors)
    assert ts.num_sites > 0
    assert ts.num_samples == 2 * meta["n_samples"]   # diploid
    y = windowed_recombination_rate(rate_map.position, rate_map.rate,
                                    meta["window_size"], meta["sequence_length"])
    assert y.shape == (200_000 // 2000,)
    assert np.all(y > 0)
    # more breakpoints where rate is higher: positive rank correlation between
    # window rate and local tree density
    bp = np.array([t.interval[0] for t in ts.trees()])[1:]
    counts, _ = np.histogram(bp, bins=np.arange(0, 200_000 + 2000, 2000))
    # Spearman-ish: compare top-rate vs bottom-rate window breakpoint counts
    order = np.argsort(y)
    lo = counts[order[:len(order)//4]].mean()
    hi = counts[order[-len(order)//4:]].mean()
    assert hi >= lo


def test_dump_one_uses_seed_in_filename_and_metadata(tmp_path):
    priors = RecombPriors(sequence_length=20_000, window_size=2_000)
    base = _dump_one(1_000_007, str(tmp_path), priors, None)
    assert base.endswith("ts_01000007")
    with np.load(base + ".npz", allow_pickle=False) as archive:
        import json

        assert json.loads(str(archive["meta"]))["seed"] == 1_000_007


def test_stdpopsim_target_is_the_exact_contig_map():
    pytest.importorskip("stdpopsim")
    ts, rate_map, meta = simulate_stdpopsim_region(
        seed=11,
        species="DroMel",
        n_samples=2,
        sequence_length=10_000,
        genetic_map=None,
        window_size=2_000,
    )
    assert ts.sequence_length == rate_map.sequence_length == 10_000
    assert ts.num_samples == 2 * meta["n_samples"]
    assert np.isclose(
        meta["mean_rate"],
        np.average(rate_map.rate, weights=np.diff(rate_map.position)),
    )
