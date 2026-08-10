"""Dog transfer figure (fig:dogtransfer) reproducibility + sanity guards.

The flagship dog figure re-plots from a committed cache (paper/figdata/dog_fig.npz) with NO
inference, so the paper build is deterministic and GPU-free (scripts/fig_dog.py). These tests guard:
  - panel-(a) LD-decay bands are physical -- mean r^2 in [0,1], one per radius. This is the exact
    class of bug that made panel (a) plot values ~5 after the 15k featurizer change reindexed the
    tokens (the old tok[:,4:4+R] read the wrong columns);
  - the transfer story holds (transfer median > own-data median);
  - the committed stats json matches the cache (the numbers the caption + claims cite);
  - the featurizer mean_r2 slice length == number of LD radii;
  - render() is byte-stable (same cache -> identical PDF).
"""
import filecmp
import os
import sys

import numpy as np
import paperlib as P
import pytest

pytestmark = pytest.mark.derived

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_ROOT, "scripts"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CACHE = P.FIGDATA / "dog_fig.npz"


def _cache():
    return dict(np.load(CACHE, allow_pickle=True))


def test_ld_decay_bands_physical():
    d = _cache()
    R = len(d["radii"])
    for k in ("vil_decay", "brd_decay"):
        v = np.asarray(d[k], float)
        assert v.shape == (R,), f"{k} shape {v.shape} != ({R},) -- one mean r^2 per radius"
        assert np.all(np.isfinite(v)) and v.min() >= 0.0 and v.max() <= 1.0 + 1e-6, \
            f"{k} not in [0,1]: {v} -- reading the wrong token columns?"


def test_transfer_beats_own():
    d = _cache()
    own, trn = np.asarray(d["own"], float), np.asarray(d["trn"], float)
    assert np.nanmedian(trn) > np.nanmedian(own), "transfer median must exceed own-data median"


def test_stats_json_matches_cache():
    d = _cache()
    s = P.figjson("dog_fig_stats")
    assert round(float(np.nanmedian(d["own"])), 4) == s["own_median"]
    assert round(float(np.nanmedian(d["trn"])), 4) == s["trn_median"]
    assert int(len(d["own"])) == s["n_regions"]


def test_mean_r2_slice_matches_radii():
    feats = pytest.importorskip("fastrho.features")
    d = _cache()
    radii = tuple(int(x) for x in np.asarray(d["radii"]).ravel())
    sl = feats.mean_r2_slice(feats.FeatureConfig(ld_radii=radii))
    assert sl.stop - sl.start == len(radii)


def test_render_is_byte_stable(tmp_path):
    pytest.importorskip("matplotlib")
    import fig_dog
    import matplotlib.pyplot as plt
    d = _cache()
    p1, p2 = tmp_path / "a.pdf", tmp_path / "b.pdf"
    fig_dog.render(d, str(p1))
    plt.close("all")
    fig_dog.render(d, str(p2))
    plt.close("all")
    assert filecmp.cmp(str(p1), str(p2), shallow=False), "render not byte-stable"
