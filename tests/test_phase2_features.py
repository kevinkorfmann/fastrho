"""Phase-2 validation: SNP-token features are well-formed AND carry recombination signal.

The signal test is the important one: with demography held constant, the local LD (r^2 to
the next SNP) must be *negatively* correlated with the true local recombination rate.
If this fails, the features are blind to recombination and no model can succeed.
"""

import msprime
import numpy as np

from fastrho.features import FeatureConfig, SNPTokenFeaturizer, feature_names, n_features
from fastrho.preprocess import _integral_fn, genotype_matrix
from fastrho.simulate import make_recombination_map


def _synthetic_gm(S=50, n=40, seed=0):
    rng = np.random.default_rng(seed)
    gm = (rng.random((n, S)) < 0.3).astype(np.int8)
    pos = np.sort(rng.uniform(0, 1e5, S))
    return gm, pos


def test_feature_shapes_and_finiteness():
    gm, pos = _synthetic_gm()
    feats = SNPTokenFeaturizer()(gm, pos, {"sequence_length": 1e5, "window_size": 2000})
    tok = feats["tokens"]
    assert tok.shape == (gm.shape[1], n_features())
    assert tok.shape[1] == len(feature_names())
    assert np.isfinite(tok).all()
    # config fractions in [0, 1]
    names = feature_names()
    for c in ("cfg_AB", "cfg_Ab", "cfg_aB", "cfg_ab"):
        col = tok[:-1, names.index(c)]            # last token has no next pair
        assert (col >= -1e-6).all() and (col <= 1 + 1e-6).all()


def test_empty_and_singleton():
    f = SNPTokenFeaturizer()
    out = f(np.zeros((10, 0), np.int8), np.zeros(0), {"sequence_length": 1e5, "window_size": 2000})
    assert out["tokens"].shape == (0, n_features())


def test_domain_randomized_feature_flags_are_unified():
    gm, pos = _synthetic_gm(S=30, n=20)
    cfg = FeatureConfig(sfs_shape=True, r2_debias=True)
    tok = SNPTokenFeaturizer(cfg)(gm, pos, {})["tokens"]
    assert tok.shape == (30, 18)
    names = feature_names(cfg)
    assert names[-1] == "local_rare_frac"
    assert np.all((tok[:, -1] >= 0) & (tok[:, -1] <= 1))
    raw = SNPTokenFeaturizer(FeatureConfig())(gm, pos, {})["tokens"]
    for radius in cfg.ld_radii:
        assert np.all(tok[:, names.index(f"mean_r2_{radius}")] <=
                      raw[:, feature_names().index(f"mean_r2_{radius}")] + 1e-7)


def test_ld_signal_correlates_with_recombination():
    """The neighbourhood-aggregated LD feature must track the scale-matched local rate."""
    from scipy.stats import spearmanr
    names = feature_names()
    R = FeatureConfig().ld_radii[-1]                 # largest radius (50 kb)
    col = names.index(f"mean_r2_{R}")
    all_f, all_rate = [], []
    L, Ne, mu = 1_000_000, 1e4, 1.5e-8
    for s in range(8):
        rm = make_recombination_map(L, np.random.default_rng(s), kind="gp", mean_rate=1e-8)
        ts = msprime.sim_ancestry(samples=50, population_size=Ne,
                                  recombination_rate=rm, sequence_length=L,
                                  random_seed=s + 1)
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=s + 101)
        gm, pos = genotype_matrix(ts)
        if len(pos) < 100:
            continue
        tok = SNPTokenFeaturizer()(gm, pos, {"sequence_length": L, "window_size": 2000})["tokens"]
        integ = _integral_fn(rm.position, rm.rate)
        lo = np.clip(pos - R / 2, 0, L)
        hi = np.clip(pos + R / 2, 0, L)
        local_rate = (integ(hi) - integ(lo)) / np.maximum(hi - lo, 1.0)
        all_f.append(tok[:, col])
        all_rate.append(local_rate)
    f = np.concatenate(all_f)
    rate = np.concatenate(all_rate)
    m = np.isfinite(f) & (f > 0) & np.isfinite(rate) & (rate > 0)
    rho, p = spearmanr(f[m], rate[m])
    print(f"Spearman(mean_r2_{R}, local rate) = {rho:.3f}  p={p:.1e}  n={m.sum()}")
    # per-token, neighbour-capped feature: a clear negative signal (the SSM sharpens it
    # across token context). Bar is "not blind to recombination", not "matches pyrho".
    assert rho < -0.25, f"aggregated LD does not track recombination (rho={rho:.3f})"
    assert p < 1e-6
