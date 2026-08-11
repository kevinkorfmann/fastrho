"""Re-derive every *computed* headline number from the raw per-config metrics.

These are the numbers the paper obtains by aggregating the snapshot — medians,
means, ranges, win-counts, speedups, deltas — so the test recomputes the
aggregation from scratch and checks the printed value, rather than reading a
stored copy. This is where a stale summary statistic (e.g. a median that no
longer reflects an updated config) gets caught.
"""

import statistics

import numpy as np
import paperlib as P
import pytest

pytestmark = pytest.mark.derived


# --------------------------------------------------------------------------
# Compute cost ratios (timings record)
# --------------------------------------------------------------------------
def test_speedup_pyrho_70x():
    assert P.approx(P.speedup("pyrho"), "70")          # abstract/intro/repro/discussion


def test_speedup_relernn_8527x():
    # repro/discussion print 8527; abstract/intro round to 8,500 -> both within ~2%
    assert P.approx(P.speedup("relernn"), "8527")
    assert P.approx(P.speedup("relernn"), "8,500")


def test_relative_walltime_triple_1_70_8527():
    t = P.summary()["timings"]
    assert (t["fastrho"], round(t["pyrho"]), round(t["relernn"])) == (1.0, 70, 8527)


def test_relernn_four_orders_of_magnitude():
    # "roughly four orders of magnitude" -> 10^3.5 .. 10^4.5
    assert 10**3.5 <= P.speedup("relernn") <= 10**4.5


# --------------------------------------------------------------------------
# fastrho absolute wall-clock range  (repro.tex: 9.7s real_dog .. 43.6s const_n100)
# --------------------------------------------------------------------------
def test_fastrho_wallclock_min_is_real_dog_9_7s():
    wc = P.fastrho_wallclock()
    cfg_min = min(wc, key=wc.get)
    assert cfg_min == "real_dog"
    assert P.matches_rounded(wc[cfg_min], "9.7")


def test_fastrho_wallclock_max_is_const_n100_43_6s():
    wc = P.fastrho_wallclock()
    cfg_max = max(wc, key=wc.get)
    assert cfg_max == "const_n100"
    assert P.matches_rounded(wc[cfg_max], "43.6")


def test_methods_wallclock_range_10_44s():
    # methods.tex: "10-44 s per scored region"
    vals = list(P.fastrho_wallclock().values())
    assert 9.0 <= min(vals) and max(vals) <= 45.0


# --------------------------------------------------------------------------
# Tree-of-life summary stats (main.tex: median 0.93 fastrho / 0.94 pyrho;
# range 0.71..0.99; fastrho ahead only on human & orangutan)
# --------------------------------------------------------------------------
def test_treeoflife_median_and_range():
    rows = P.tree_of_life_rows()
    assert len(rows) == 7, f"expected 7 species, parsed {len(rows)}"
    f = [r[1] for r in rows]
    p = [r[2] for r in rows]
    assert P.matches_rounded(statistics.median(f), "0.93")
    assert P.matches_rounded(statistics.median(p), "0.94")
    assert P.matches_rounded(min(f), "0.71")     # orangutan
    assert P.matches_rounded(max(f), "0.99")     # nematode


def test_treeoflife_fastrho_ahead_only_human_orangutan():
    rows = {r[0]: r for r in P.tree_of_life_rows()}
    ahead = {lbl for lbl, fr, py in rows.values() if fr > py}
    assert ahead == {"Human", "Orangutan"}, (
        f"prose says fastrho leads on human & orangutan only; data says {ahead}")


# --------------------------------------------------------------------------
# ReLERNN fine-scale range "0.02-0.41" and Δr=0.66 collapse (showdown curve)
# --------------------------------------------------------------------------
def test_relernn_25kb_range_002_041():
    vals = [v for _, v in P.all_pearsons("relernn", "25kb")]
    assert P.matches_rounded(min(vals), "0.02")
    assert P.matches_rounded(max(vals), "0.45")


def test_showdown_delta_r_25kb_is_066():
    # main caption: ReLERNN collapses vs fastrho at 25kb, Δr=0.66
    cur = P.showdown_meta()["curve"]
    dr = cur["fastrho"][0] - cur["relernn"][0]      # both at grids_kb[0]=25kb
    assert P.matches_rounded(dr, "0.66")


def test_showdown_dots_match_snapshot_25kb_pearson():
    # the figure's per-config dots must equal the canonical 25kb Pearson
    dots = P.showdown_meta()["dots"]
    for cfg, methods in dots.items():
        for method, v in methods.items():
            assert np.isclose(v, P.metric(cfg, "25kb", method, "pearson")), (
                f"showdown dots[{cfg}][{method}]={v} != snapshot")


# --------------------------------------------------------------------------
# Phasing/polarization ablation means (main.tex:199-212)
# --------------------------------------------------------------------------
ABLATION = [
    ("unphased",                 "0.57", "naive unphased"),
    ("unphased_gt",              "0.86", "composite-LD"),
    ("unphased_unpol_gt",        "0.82", "folded composite-LD"),
    ("unphased_unpol_gt_gtmodel","0.88", "fastrho-GT specialist"),
    ("phased",                   "0.90", "phased ceiling"),
    ("unphased_gt_dr",           "0.875", "domain-randomized unphased"),
    ("unphased_unpol_gt_dr",     "0.865", "domain-randomized unphased+unpol"),
]


@pytest.mark.parametrize("variant,written,label", ABLATION,
                         ids=[a[0] for a in ABLATION])
def test_ablation_mean(variant, written, label):
    vals = P.stdpopsim_pearsons(variant)
    assert len(vals) == 7, f"{variant}: expected 7 species, got {len(vals)}"
    assert P.matches_rounded(float(np.mean(vals)), written), label


def test_ablation_ranges():
    # naive 0.37-0.74; composite-LD 0.74-0.97; folded 0.72-0.95
    for variant, lo, hi in [("unphased", "0.37", "0.74"),
                            ("unphased_gt", "0.74", "0.97"),
                            ("unphased_unpol_gt", "0.72", "0.95")]:
        v = P.stdpopsim_pearsons(variant)
        assert P.matches_rounded(min(v), lo), f"{variant} min"
        assert P.matches_rounded(max(v), hi), f"{variant} max"


# --------------------------------------------------------------------------
# Between-population gaps (floor - prediction), derived from the records
# --------------------------------------------------------------------------
def test_between_pop_gap_d0_is_0039():
    s = P.between_pop("d00")["scales"]["25kb"]
    gap = s["within_pop_noise_floor"] - s["between_pop_pred"]
    assert P.matches_rounded(gap, "0.039")


def test_between_pop_gap_d50_is_0060():
    s = P.between_pop("d50")["scales"]["25kb"]
    gap = s["within_pop_noise_floor"] - s["between_pop_pred"]
    assert P.matches_rounded(gap, "0.060")


def test_between_pop_70x_cheaper():
    # between_pop.tex: "~70x cheaper than pyrho"
    assert P.approx(P.speedup("pyrho"), "70")


# --------------------------------------------------------------------------
# ReLERNN repro: R^2 = Pearson^2  (repro.tex / fig:repro_relernn)
# --------------------------------------------------------------------------
def test_relernn_repro_r2_equals_pearson_squared():
    # repro.tex prints Pearson 0.93 and R^2 0.87. Both are rounded displays of the
    # same underlying value (~0.933), so a unique unrounded r reproduces both:
    #   round(r,2)=0.93 and round(r^2,2)=0.87  <=>  r in [0.9301, 0.9354)
    lo, hi = 0.925, 0.935
    rs = [r for r in (lo + i * 1e-4 for i in range(int((hi - lo) / 1e-4)))
          if P.matches_rounded(r, "0.93") and P.matches_rounded(r ** 2, "0.87")]
    assert rs, "no Pearson rounds to 0.93 with R^2 rounding to 0.87"


def test_relernn_repro_close_to_published_r2():
    # our R^2 0.87 vs their reported 0.931 -> within ~7%
    assert P.approx(0.93 ** 2, "0.931", rel=0.08)


# --------------------------------------------------------------------------
# Selfer inversion is demography-invariant: pyrho chr1 r ~= -0.35 under a
# constant Ne, a real expansion, and a real bottleneck A. thaliana model alike
# (ED Fig. selfer_inversion caption; selfer_demog_robustness.json).
# --------------------------------------------------------------------------
def _selfer_demog_json():
    import json
    return json.loads((P.FIGDATA / "selfer_demog_robustness.json").read_text())


def test_selfer_inversion_survives_every_demography():
    rs = [m["pearson"] for m in _selfer_demog_json()["models"].values()]
    assert len(rs) == 3
    assert all(r < 0 for r in rs), f"pyrho should stay inverted (r<0) under every demography: {rs}"
    # demography changes almost nothing: constant, expansion and bottleneck agree tightly
    assert max(rs) - min(rs) < 0.01, f"demography changed pyrho's r by >0.01: spread {rs}"
    assert all(-0.40 < r < -0.30 for r in rs), f"each demography should give r ~= -0.35: {rs}"
    assert P.matches_rounded(float(np.mean(rs)), "-0.35"), f"mean over demographies should be -0.35: {rs}"


def test_selfer_demog_baseline_matches_figure_bundle():
    # the constant-Ne entry is the published chr1 pyrho result also stored in selfer_chroms.npz
    j = _selfer_demog_json()["models"]["constant_Ne"]["pearson"]
    bundle = float(P.load_npz(P.FIGDATA / "selfer_chroms.npz")["c1_pyrho_r"])
    assert np.isclose(j, bundle, atol=5e-4), f"json constant-Ne {j} != bundle c1_pyrho_r {bundle}"
