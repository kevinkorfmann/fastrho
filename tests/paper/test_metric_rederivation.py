"""Gold-standard re-derivation: recompute the reported metrics directly from the
committed raw (prediction, truth) arrays, using the same estimator the pipeline
used (paperlib.score_rates, a byte-for-byte mirror of fastrho.evaluate). If a
stored metric and the array it came from disagree, the snapshot itself is wrong
— a stronger guarantee than table/prose consistency.
"""

import numpy as np
import paperlib as P
import pytest

pytestmark = pytest.mark.rederive


# --------------------------------------------------------------------------
# Mirror sanity: our score_rates must equal fastrho.evaluate.score_rates.
# --------------------------------------------------------------------------
def test_score_rates_mirrors_the_package():
    try:
        from fastrho.evaluate import score_rates as pkg_score
    except Exception as e:                       # pragma: no cover
        pytest.skip(f"fastrho.evaluate not importable: {e}")
    rng = np.random.default_rng(0)
    pred = rng.lognormal(-18, 1, 500)
    true = pred * rng.lognormal(0, 0.3, 500)
    a, b = P.score_rates(pred, true), pkg_score(pred, true)
    assert a.keys() == b.keys()
    for k in a:
        assert np.isclose(a[k], b[k]), f"{k}: {a[k]} != {b[k]}"


# --------------------------------------------------------------------------
# Dipteran bias region (figdata/dipteran_bias.npz): recompute the embedded stats
# from rhat (pred) vs rtrue (truth) + the hotspot mask.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dipteran():
    import json
    z = P.load_npz(P.FIGDATA / "dipteran_bias.npz")
    stats = json.loads(str(z["stats"]))
    return z, stats


def test_dipteran_logpearson_rederived(dipteran):
    # integrity: the stored per-SNP log-Pearson re-derives from the committed arrays.
    # (Not a headline prose number for the high-Ne-fixed model; the landscape claim is the
    #  windowed Pearson in summary.json -- Drosophila 0.99, Anopheles 0.91.)
    from scipy.stats import pearsonr
    z, stats = dipteran
    pred, true = np.asarray(z["rhat"], float), np.asarray(z["rtrue"], float)
    m = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    lp = float(pearsonr(np.log(pred[m]), np.log(true[m]))[0])
    assert np.isclose(lp, stats["logpearson"], atol=1e-6)


def test_dipteran_spearman_rederived(dipteran):
    from scipy.stats import spearmanr
    z, stats = dipteran
    pred, true = np.asarray(z["rhat"], float), np.asarray(z["rtrue"], float)
    m = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    sp = float(spearmanr(pred[m], true[m])[0])
    assert np.isclose(sp, stats["spearman"], atol=1e-6)


def test_dipteran_baseline_bias_rederived(dipteran):
    """The high-Ne-fixed model recovers the absolute background directly: the baseline
    (non-hotspot) bias ratio (~0.78x on this region, the Fig.(a) value, up from the earlier
    model's 0.45x) reproduces exactly from the committed arrays, and the residual genome-mean
    anchor brings it to ~1.0x."""
    z, stats = dipteran
    pred, true = np.asarray(z["rhat"], float), np.asarray(z["rtrue"], float)
    hot = np.asarray(z["hot"], bool)
    m = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    base = float(np.median(pred[m & ~hot] / true[m & ~hot]))
    assert np.isclose(base, stats["baseline_br"], atol=1e-6)
    assert P.matches_rounded(base, "0.82")                      # Fig. dipteran (a) value
    # residual genome-mean anchor restores absolute scale to ~1.0x
    anchored = base * stats["anchor_scale"]
    assert np.isclose(anchored, stats["baseline_br_anchored"], atol=1e-6)
    assert P.matches_rounded(anchored, "1.0")
    # if any interval is hotspot-classified, the stored split is finite and ordered sensibly
    # (this smooth high-Ne region may have none -> hotspot_br is nan, which is fine).
    if np.isfinite(stats["hotspot_br"]):
        assert stats["hotspot_br"] <= stats["baseline_br"] * 1.5


def test_dipteran_frac_hot_rederived(dipteran):
    z, stats = dipteran
    assert np.isclose(float(np.mean(np.asarray(z["hot"], bool))), stats["frac_hot"], atol=1e-9)


# --------------------------------------------------------------------------
# Selection hero region (figdata/selection_figdata.npz): recompute hero_pearson.
# --------------------------------------------------------------------------
def test_selection_hero_pearson_rederived():
    z = P.load_npz(P.FIGDATA / "selection_figdata.npz")
    pred = np.asarray(z["hero_fastrho"], float)
    true = np.asarray(z["hero_truth"], float)
    r = P.pearson(pred, true)
    assert np.isclose(r, float(z["hero_pearson"]), atol=1e-6)


# --------------------------------------------------------------------------
# Showdown geometry (figdata/relernn_showdown.npz): hotspot peak/location and
# ReLERNN's flat-block width and rate that the paper quotes.
# --------------------------------------------------------------------------
def test_showdown_relernn_block_is_82kb_and_4e8():
    z = P.load_npz(P.FIGDATA / "relernn_showdown.npz")
    meta = P.showdown_meta()
    edges = np.asarray(z["rel_edges"], float)
    rates = np.asarray(z["rel_rates"], float)
    widths_kb = np.diff(edges) / 1000.0
    assert P.matches_rounded(float(np.median(widths_kb)), "82")        # block width
    # the block covering the hotspot location reports a single ~4e-8 rate
    hc = float(meta["hc"])
    bi = int(np.searchsorted(edges, hc) - 1)
    assert P.approx(float(rates[bi]), "4e-8", rel=0.10)


def test_showdown_hotspot_peak_and_location():
    meta = P.showdown_meta()
    assert P.approx(float(meta["hotspot_true"]), "7.1e-7")
    assert P.approx(float(meta["hc"]) / 1000.0, "824")
