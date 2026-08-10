"""Real between-population recombination divergence, separated from inference noise.

The paper's sharpest capability, on real data. A point-estimate method (pyrho, ReLERNN)
can only report ONE number -- how well two populations' inferred maps correlate. It cannot
say how much of the shortfall from 1.0 is genuine biological divergence versus finite-sample
inference noise. fastrho draws the *noise floor* directly: infer each population's map on two
DISJOINT half-samples and correlate -- that is what "identical populations, this much data"
looks like. The gap between that floor and the real CEU-vs-YRI correlation is divergence you
can actually believe.

(a) The money panel: on real 1000G humans (12 loci, 25/100/500 kb), the within-population noise
    floor and the CEU-vs-YRI between-population correlation, with the excess-divergence gap
    shaded. The gap shrinks toward broad scale -- the fine-scale hotspot-turnover signature.
    Point estimates cannot draw the grey line at all.
(b) The PRDM9 gradient. Excess divergence below the noise floor for two clades in opposite
    regimes: PRDM9+ humans diverge genome-wide by hotspot turnover (~0.22); PRDM9- Anopheles
    are conserved in collinear genome (~0.07-0.08) yet diverge sharply INSIDE the 2La inversion
    (~0.43).
(c) External validation: fastrho independently recovers each population's PUBLISHED pyrho map
    (matched > cross), so the maps compared in (a) are real, not artifacts.

House style via paper_style (locked palette, house fonts, no bold titles). Reads only committed
JSON in paper/figdata/. Build:  python3.13 scripts/fig_between_pop_real.py
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "scripts")
import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDATA = os.path.join(HERE, "paper", "figdata")
FIG = os.path.join(HERE, "paper", "figures")

# ---- locked palette -----------------------------------------------------------------
BLUE, LBLUE = ps.C["fastrho"], ps.C["fastrho_l"]     # dark / light blue
GREY, BLACK = ps.C["relernn"], ps.C["truth"]
RED = ps.CB[4]                                         # ColorBrewer "Paired" red #e31a1c
ORANGE = ps.CB[5]                                      # #ff7f00
BAND = "#eef1f4"                                       # low-chroma neutral wash
GAP = "#cfe3f1"                                        # light-blue gap fill (= the measured signal)

TITLE_FS = 9.7

ps.style()
plt.rcParams.update({"grid.alpha": 0.18})


def load(name):
    return json.load(open(os.path.join(FIGDATA, name)))


def main():
    hum = load("real_between_pop.json")
    agam = load("agam_noise_floor.json")
    val = load("real_between_pop_pyrho.json")

    # ---- panel (a) data: scale-dependent floor vs between (sample-size MATCHED) -------
    order = ["25kb", "100kb", "500kb"]
    sc = hum["scales"]
    floor = np.array([sc[k]["within_floor_mean"] for k in order])
    betw = np.array([sc[k]["between_matched_n"] for k in order])
    drop = np.array([sc[k]["excess_drop_matched"] for k in order])
    auc = hum["differential_hotspots_25kb"]["separability_auc_between_vs_noise"]

    # ---- panel (b) data: excess divergence below the noise floor, by clade/regime ----
    a3l = agam["arms"]["3L"]
    a2l = agam["arms"]["2L"]
    hum_excess = sc["25kb"]["excess_drop_matched"]
    mos_3l = a3l["within_floor_mean"] - a3l["between_mean"]
    mos_2l_out = a2l["within_floor_mean"] - a2l["outside_2la_between"]
    mos_2l_in = a2l["within_floor_mean"] - a2l["inside_2la_between"]

    # ---- panel (c) data: fastrho vs published pyrho, matched vs cross ----------------
    v = val["pooled"]["validation_25kb"]
    ceu_match, ceu_cross = v["fastrhoCEU_vs_pyrhoCEU"], v["fastrhoCEU_vs_pyrhoYRI"]
    yri_match, yri_cross = v["fastrhoYRI_vs_pyrhoYRI"], v["fastrhoYRI_vs_pyrhoCEU"]
    pyrho_between = v["pyrhoCEU_vs_pyrhoYRI"]

    print("(a) floor  :", np.round(floor, 3))
    print("(a) between:", np.round(betw, 3))
    print("(a) drop   :", np.round(drop, 3), "| AUC", round(auc, 2))
    print("(b) excess : human %.2f | 3L %.3f | 2L-out %.3f | 2L-in %.3f"
          % (hum_excess, mos_3l, mos_2l_out, mos_2l_in))
    print("(c) CEU match/cross %.3f/%.3f | YRI match/cross %.3f/%.3f | pyrho-between %.3f"
          % (ceu_match, ceu_cross, yri_match, yri_cross, pyrho_between))

    fig = plt.figure(figsize=(13.9, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.12, 0.98], wspace=0.40,
                          left=0.062, right=0.985, bottom=0.155, top=0.845)

    # ==================================================================== (a)
    a = fig.add_subplot(gs[0])
    x = np.arange(3)
    a.set_xlim(-0.32, 2.32)
    a.set_ylim(0.598, 1.028)

    # "identical maps" ceiling (what point estimates implicitly hope for)
    a.axhline(1.0, color="#c9c9c9", lw=1.0, ls=(0, (4, 3)), zorder=1)
    a.text(2.30, 1.006, "identical maps", ha="right", va="bottom", fontsize=7.4, color="#9a9a9a")

    # the excess-divergence gap = the quantity only fastrho can isolate
    a.fill_between(x, betw, floor, color=GAP, alpha=0.95, zorder=1.5,
                   label="excess divergence (real, above noise)")

    # noise floor (grey) -- disjoint half-samples of the SAME population
    a.plot(x, floor, "-o", color=GREY, lw=2.6, ms=8.5, mec="white", mew=1.4, zorder=4,
           label="within-population noise floor (half vs half)")
    # CEU vs YRI between-population (blue)
    a.plot(x, betw, "-o", color=BLUE, lw=2.6, ms=8.5, mec="white", mew=1.4, zorder=4,
           label="CEU vs YRI (between-population)")

    # annotate the shrinking gap: double arrow + value at each window
    for xi, lo, hi, d in zip(x, betw, floor, drop):
        a.annotate("", xy=(xi, hi), xytext=(xi, lo),
                   arrowprops=dict(arrowstyle="<->", color="#3f6f96", lw=1.15), zorder=5)
        a.text(xi + 0.085, (lo + hi) / 2, "%.2f" % d, ha="left", va="center",
               fontsize=9.0, color=BLUE, fontweight="bold", zorder=6)

    # narrative: gap shrinks toward broad scale
    a.annotate("gap shrinks toward\nbroad scale = fine-scale\nhotspot turnover",
               xy=(2.0, (betw[2] + floor[2]) / 2), xytext=(1.18, 0.735),
               textcoords="data", fontsize=7.9, color="#3f6f96", ha="left", va="center",
               arrowprops=dict(arrowstyle="->", color="#7ba3c4", lw=1.0,
                               connectionstyle="arc3,rad=-0.25"))
    # the capability line: point estimates report only the blue value
    a.text(0.0, 0.663, "point estimates report only the blue value — with no floor, "
           "0.70 is uninterpretable", fontsize=7.6, color="#666", ha="left", va="center")

    a.set_xticks(x)
    a.set_xticklabels(["25 kb", "100 kb", "500 kb"])
    a.set_xlabel(r"scoring window  (fine $\rightarrow$ broad)")
    a.set_ylabel("map correlation (Pearson $r$)")
    a.legend(loc="upper left", bbox_to_anchor=(0.005, 0.995), fontsize=8.0,
             handlelength=1.5, labelspacing=0.35, borderaxespad=0.0)
    a.set_title("(a) real CEU–YRI divergence, separated from inference noise",
                fontsize=TITLE_FS, loc="left")
    ps.panel(a, "a", x=-0.115, y=1.11)

    # ==================================================================== (b)
    b = fig.add_subplot(gs[1])
    xs = np.array([0.0, 1.35, 2.05, 2.75])
    vals = np.array([hum_excess, mos_3l, mos_2l_out, mos_2l_in])
    cols = [BLUE, GREY, GREY, RED]
    labs = ["genome-wide", "3L\ncollinear", "2L outside\n2La", "2L inside\n2La"]
    b.set_xlim(-0.5, 3.25)
    b.set_ylim(0, 0.49)

    # "conserved" reference band (no hotspot turnover)
    b.axhspan(0, 0.10, color=BAND, zorder=0)
    b.text(3.20, 0.088, "conserved", ha="right", va="top", fontsize=7.4, color="#9aa0a6")

    b.vlines(xs, 0, vals, color="#d9d9d9", lw=3.0, zorder=2)
    b.scatter(xs, vals, s=185, c=cols, edgecolor=BLACK, linewidth=0.9, zorder=4)
    for xi, vi, ci in zip(xs, vals, cols):
        b.annotate("%.2f" % vi, (xi, vi), xytext=(0, 8), textcoords="offset points",
                   ha="center", va="bottom", fontsize=8.6, color=ci, fontweight="bold")

    b.set_xticks(xs)
    b.set_xticklabels(labs, fontsize=8.2)
    b.set_ylabel("excess divergence below noise floor")

    # clade group brackets
    def clade(x0, x1, txt, col):
        yb = -0.122
        b.plot([x0, x0, x1, x1], [yb + 0.018, yb, yb, yb + 0.018], color=col, lw=1.2,
               clip_on=False, transform=b.transData)
        b.text((x0 + x1) / 2, yb - 0.012, txt, ha="center", va="top", fontsize=8.2,
               color=col, clip_on=False, transform=b.transData)
    clade(-0.28, 0.28, "humans\n(PRDM9+)", BLUE)
    clade(1.07, 3.03, "mosquitoes (PRDM9−)", "#8a5a2b")

    # regime annotations
    b.annotate("hotspot\nturnover", (xs[0], hum_excess), xytext=(11, -6),
               textcoords="offset points", fontsize=7.6, color=BLUE, ha="left", va="top")
    b.annotate("inversion\nsuppresses\nexchange", (xs[3], mos_2l_in), xytext=(-8, -2),
               textcoords="offset points", fontsize=7.6, color=RED, ha="right", va="top")

    b.set_title("(b) the PRDM9 gradient: turnover vs conserved-except-inversions",
                fontsize=TITLE_FS, loc="left")
    ps.panel(b, "b", x=-0.155, y=1.11)

    # ==================================================================== (c)
    c = fig.add_subplot(gs[2])
    groups = np.array([0.0, 1.0])
    w = 0.34
    match = np.array([ceu_match, yri_match])
    cross = np.array([ceu_cross, yri_cross])
    c.set_ylim(0.0, 1.0)
    c.set_xlim(-0.55, 1.55)

    bm = c.bar(groups - w / 2, match, w, color=BLUE, edgecolor="white", linewidth=0.6,
               label="matched population", zorder=3)
    bc = c.bar(groups + w / 2, cross, w, color=LBLUE, edgecolor="white", linewidth=0.6,
               label="cross population", zorder=3)
    for rect in list(bm) + list(bc):
        h = rect.get_height()
        c.annotate("%.2f" % h, (rect.get_x() + rect.get_width() / 2, h),
                   xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                   fontsize=7.8, color="#444")

    # published pyrho CEU-vs-YRI: a point estimate with NO floor to interpret it
    c.axhline(pyrho_between, color=GREY, lw=1.1, ls=(0, (4, 3)), zorder=2)
    c.text(1.52, pyrho_between + 0.008, "pyrho CEU vs YRI = %.2f\n(point estimate, no floor)"
           % pyrho_between, ha="right", va="bottom", fontsize=7.0, color="#7a7a7a")

    c.set_xticks(groups)
    c.set_xticklabels(["fastrho CEU\nvs pyrho", "fastrho YRI\nvs pyrho"], fontsize=8.4)
    c.set_ylabel("correlation with published pyrho ($r$)")
    c.legend(loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=7.8,
             handlelength=1.2, ncol=1, labelspacing=0.3)
    c.set_title("(c) validation: each population's own pyrho map recovered",
                fontsize=TITLE_FS, loc="left")
    ps.panel(c, "c", x=-0.20, y=1.11)

    # small supporting note in (a): differential hotspots are separable from noise
    a.text(0.0, 0.622, "differential hotspots separable from noise:  AUC = %.2f" % auc,
           ha="left", va="center", fontsize=7.2, color="#888")

    out = os.path.join(FIG, "fig_between_pop_real.pdf")
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
