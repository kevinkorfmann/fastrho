"""Structural sanity of the canonical results JSON.

Before we trust the snapshot as the source of every paper number, we assert it
is internally well-formed: metrics live in their valid ranges, interval counts
fall monotonically as the scoring window coarsens, and the cost record is
ordered fastrho < pyrho < ReLERNN.
"""

import paperlib as P
import pytest

pytestmark = pytest.mark.integrity


# ---- every benchmark config carries its design metadata -------------------
@pytest.mark.parametrize("config", P.BENCH_CONFIGS)
def test_config_has_metadata(config):
    rec = P.summary()[config]
    for key in ("config", "demography", "n_hap", "Ne", "n_regions_scored", "scales"):
        assert key in rec, f"{config} missing {key!r}"
    assert rec["n_hap"] > 0 and rec["Ne"] > 0
    assert rec["n_regions_scored"] > 0


# ---- enumerate every present (config, scale, method) metric cell ----------
def _present_cells():
    cells = []
    for c in P.BENCH_CONFIGS:
        scales = P.summary()[c].get("scales", {})
        for s, methods in scales.items():
            for m, vec in methods.items():
                if isinstance(vec, dict) and "pearson" in vec:
                    cells.append((c, s, m))
    return cells


CELLS = _present_cells()
CELL_IDS = [f"{c}-{s}-{m}" for c, s, m in CELLS]


@pytest.mark.parametrize("config,scale,method", CELLS, ids=CELL_IDS)
def test_metric_ranges(config, scale, method):
    cell = P.summary()[config]["scales"][scale][method]
    for corr in ("pearson", "spearman", "log_pearson"):
        if corr in cell:
            assert -1.0 <= cell[corr] <= 1.0, f"{corr}={cell[corr]} out of [-1,1]"
    if "hotspot_auprc" in cell:
        assert 0.0 <= cell["hotspot_auprc"] <= 1.0
    for pos in ("l2", "log_l2", "bias_ratio"):
        if pos in cell:
            assert cell[pos] > 0, f"{pos} must be > 0, got {cell[pos]}"
    assert cell["n"] >= 3, "score_rates needs >=3 paired points to report metrics"


# ---- interval count must fall as the window widens ------------------------
@pytest.mark.parametrize("config,method", sorted({(c, m) for c, _, m in CELLS}),
                         ids=lambda v: "-".join(v) if isinstance(v, tuple) else str(v))
def test_n_monotone_in_scale(config, method):
    ns = {}
    for s in P.SCALES:
        try:
            ns[s] = P.metric(config, s, method, "n")
        except KeyError:
            pass
    present = [ns[s] for s in P.SCALES if s in ns]
    assert present == sorted(present, reverse=True), (
        f"{config}/{method}: scored-interval count should be non-increasing "
        f"from 25kb->100kb->500kb, got {ns}")


# ---- cost record ordering -------------------------------------------------
def test_timings_ordering():
    t = P.summary()["timings"]
    assert t["fastrho"] == 1.0, "fastrho is the normalization baseline (==1)"
    assert t["pyrho"] > t["fastrho"]
    assert t["relernn"] > t["pyrho"]


# ---- held-out calibration record is well-formed ---------------------------
def test_heldout_record_shape():
    h = P.summary()["heldout"]["coverage_curve"]
    nom, emp = h["nominal"], h["empirical"]
    assert len(nom) == len(emp) > 0
    assert nom == sorted(nom) and emp == sorted(emp), "curves must be monotone"
    assert all(0 < x < 1 for x in nom + emp)
    assert P.summary()["heldout"]["n_intervals"] > 0
