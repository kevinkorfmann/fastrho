"""Figure 5 -- 'One architecture reads every data type' (unphased / unpolarized robustness).

Real-maps-forward redesign (supersedes the old three-panel version):
  (a) One model, every data type. As the data degrade (phased -> unphased -> unphased+unpolarized)
      the naive haplotype featurizer collapses (0.90 -> 0.57), but a SINGLE domain-randomized (DR)
      network holds near the phased ceiling (0.875, 0.865). This is the title claim, shown as two
      diverging trajectories, and it absorbs the one useful point of the old rescue slopegraph
      (the collapse that is avoided). The per-condition GT specialist is a faint reference.
  (b) The proof: real true-vs-DR-predicted recombination maps recovered from unphased+unpolarized
      genotypes, for six species across the tree of life (human/baboon PRDM9 hotspots, dog
      centromeric suppression, fly gene-dense, nematode arm-step, thale-cress pericentromere dip),
      in a 3x2 gallery with DR 95% prediction bands.

The old panel (b) -- the composite-LD rescue slopegraph on the frozen base model -- is dropped from
the main figure; its numbers (naive 0.57 -> composite-LD 0.86 -> folded 0.82, no retraining) live in
the text and are summarised by panel (a).

Reads paper/figures/_stdpopsim_*.json + _stdpopsim_panelc_track10.json + the silhouettes.
Run on the mac:  python scripts/fig_unphased.py
"""
from __future__ import annotations
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "paper", "figures")
SILDIR = os.path.join(FIGDIR, "silhouettes")
ORDER = ["human", "orangutan", "baboon", "dog", "fly", "worm", "arabidopsis"]
MODES = {"phased": "phased", "naive": "unphased", "gt": "unphased_gt",
         "gtfold": "unphased_unpol_gt", "spec": "unphased_unpol_gt_gtmodel",
         "dr_pol": "unphased_gt_dr", "dr_unpol": "unphased_unpol_gt_dr"}

RAW = {sk: json.load(open(os.path.join(FIGDIR, f"_stdpopsim_{m}.json"))) for sk, m in MODES.items()}
P = {k: {sk: RAW[sk][k]["pearson"] for sk in MODES} for k in ORDER}
COMMON = {k: RAW["phased"][k]["common"] for k in ORDER}
MEANS = {sk: float(np.mean([P[k][sk] for k in ORDER])) for sk in MODES}

ps.style()
C = ps.C
BLUE, LIGHT, GREY, INK, SUB, TRUTH = (
    C["fastrho"], C["relernn"], "#8A8A8A", "#151515", "#666666", C["truth"]
)

_SIL: dict[str, np.ndarray] = {}
def _sil(key):
    if key not in _SIL:
        _SIL[key] = mpimg.imread(os.path.join(SILDIR, key + ".png"))
    return _SIL[key]

def add_sil(ax, key, loc, target_pts=30, alpha=0.85, box_alignment=(1, 1)):
    arr = _sil(key)
    oi = OffsetImage(arr, zoom=target_pts / max(arr.shape[0], arr.shape[1]), alpha=alpha)
    ax.add_artist(AnnotationBbox(oi, loc, xycoords="axes fraction",
                  box_alignment=box_alignment, frameon=False, pad=0, zorder=20))


# ======================================================================== figure
fig = plt.figure(figsize=(7.15, 6.25))
ax = fig.add_axes([0.100, 0.830, 0.855, 0.120])                    # (a) data-type sweep (top strip)
# (b) 3x2 gallery of real recovered maps
_GX = (0.085, 0.560); _GY = (0.545, 0.310, 0.075); _GW, _GH = 0.395, 0.165
cax = [fig.add_axes([gx, gy, _GW, _GH]) for gy in _GY for gx in _GX]  # row-major: TL,TR,ML,MR,BL,BR

# ------------------------------------------------------------- panel (a): one model, every type
ceilv = MEANS["phased"]
X = [0, 1, 2]
ax.axhline(ceilv, color=BLUE, ls=(0, (5, 2.4)), lw=1.1, alpha=0.6, zorder=1)
ax.text(0.02, ceilv + 0.009, f"phased ceiling  ({ceilv:.2f})", fontsize=7.4, color=BLUE,
        va="bottom", ha="left", alpha=0.95)

# naive: correct on phased, collapses on unphased
ax.plot([0, 1], [ceilv, MEANS["naive"]], color=GREY, lw=1.5, ls="--", zorder=4,
        solid_capstyle="round")
for k in ORDER:
    ax.scatter([1], [P[k]["naive"]], s=11, color=GREY, alpha=0.30, lw=0, zorder=3)
ax.scatter([1], [MEANS["naive"]], marker="X", s=95, color=GREY, edgecolor="white", lw=1.0, zorder=6)
ax.annotate("naïve haplotype\nfeaturizer collapses", (1, MEANS["naive"]), xytext=(1.16, 0.60),
            fontsize=7.4, color="#555", va="center", ha="left",
            arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))
ax.text(1, MEANS["naive"] - 0.028, f"{MEANS['naive']:.2f}", ha="center", va="top",
        fontsize=7.8, color="#555")

# DR: one model across every data type, near the ceiling
dry = [ceilv, MEANS["dr_pol"], MEANS["dr_unpol"]]
for k in ORDER:                                                    # per-species spread
    ax.scatter([1, 2], [P[k]["dr_pol"], P[k]["dr_unpol"]], s=11, color=BLUE, alpha=0.28, lw=0, zorder=5)
ax.plot(X, dry, color=BLUE, lw=1.5, zorder=7, solid_capstyle="round")
ax.scatter([0], [ceilv], s=60, facecolor="white", edgecolor=BLUE, lw=1.6, zorder=8)
ax.scatter([1, 2], dry[1:], s=88, color=BLUE, edgecolor="white", lw=1.3, zorder=9)
for xi, yi in zip([1, 2], dry[1:]):
    ax.text(xi, yi + 0.018, f"{yi:.2f}", ha="center", va="bottom", fontsize=9, color=BLUE,
            fontweight="medium")
ax.text(1.5, 0.845, "one DR model", fontsize=8.1, color=BLUE, style="italic", ha="center",
        va="top", rotation=-3)

# GT specialist: per-condition reference (best single-condition model)
ax.scatter([2.16], [MEANS["spec"]], marker="D", s=42, color=LIGHT, edgecolor="white", lw=0.8, zorder=6)
ax.annotate("GT specialist\n(best per-condition)", (2.16, MEANS["spec"]), xytext=(2.30, 0.80),
            fontsize=6.9, color="#5b7ea6", va="center", ha="left",
            arrowprops=dict(arrowstyle="->", color=LIGHT, lw=0.9))

ax.set_xticks(X); ax.set_xticklabels(["phased", "unphased", "unphased +\nunpolarized"], fontsize=8.0)
ax.set_xlim(-0.35, 3.05); ax.set_ylim(0.50, 0.95)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylabel("map accuracy\n(Pearson $r$)", fontsize=8.3, labelpad=3)
ax.tick_params(axis="y", labelsize=7.4)
ax.grid(axis="y", color="#b8b8b8", alpha=0.28, lw=0.6); ax.grid(axis="x", visible=False)
ax.tick_params(axis="x", length=0)
ax.spines["bottom"].set_color("#888"); ax.spines["left"].set_color("#888")
fig.text(0.028, 0.982, "a", fontsize=10, fontweight="bold", va="top", ha="left")
fig.text(0.100, 0.982, "one architecture reads every data type", fontsize=8.8,
         style="italic", color=SUB,
         va="top", ha="left")

# ------------------------------------------------------------- panel (b): real recovered maps
T10 = json.load(open(os.path.join(FIGDIR, "_stdpopsim_panelc_track10.json")))
ARCH = {"human": "PRDM9 hotspots", "baboon": "PRDM9 hotspots", "dog": "centromeric suppression",
        "fly": "gene-dense (no PRDM9)", "worm": "chromosome-arm step",
        "arabidopsis": "pericentromere dip"}
GALLERY = ["human", "baboon", "dog", "fly", "worm", "arabidopsis"]   # row-major TL,TR,ML,MR,BL,BR
for gi, (k, axc) in enumerate(zip(GALLERY, cax)):
    t = T10[k]
    x = np.asarray(t["centers"], float)
    tr = np.asarray(t["truth"], float); pr = np.asarray(t["pred"], float)
    lo = np.asarray(t["ci_lo"], float); hi = np.asarray(t["ci_hi"], float)
    pmin = np.concatenate([tr[tr > 0], pr[pr > 0], lo[lo > 0]]).min(); b0 = pmin * 0.5
    tr = np.where(tr > 0, tr, b0); pr = np.where(pr > 0, pr, b0)
    lo = np.where(lo > 0, lo, b0); hi = np.where(hi > 0, hi, b0)
    axc.plot(x, lo, color=BLUE, lw=0.55, ls=(0, (2, 2)), alpha=0.55, zorder=2)
    axc.plot(x, hi, color=BLUE, lw=0.55, ls=(0, (2, 2)), alpha=0.55, zorder=2)
    axc.plot(x, tr, color=TRUTH, lw=1.3, ls=(0, (4, 2.5)), zorder=4, solid_capstyle="round")
    axc.plot(x, pr, color=BLUE, lw=1.5, alpha=0.95, zorder=5, solid_capstyle="round")
    axc.set_yscale("log"); axc.set_xlim(x.min(), x.max())
    axc.set_ylim(b0 * 0.8, max(tr.max(), pr.max(), hi.max()) * 2.0)
    axc.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    axc.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    axc.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto", numticks=12))
    axc.yaxis.set_minor_formatter(NullFormatter())
    axc.tick_params(axis="y", which="major", length=2.4, labelsize=7.4)
    axc.tick_params(axis="y", which="minor", length=0)
    axc.grid(False)
    xmx = x.max(); axc.set_xticks([0, round(xmx / 2), round(xmx)])
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.spines["left"].set_color("#c0c0c0"); axc.spines["bottom"].set_color("#9a9a9a")
    add_sil(axc, k, (0.985, 0.95), target_pts=25, alpha=0.85, box_alignment=(1, 1))
    axc.text(0.015, 0.94, f"{COMMON[k]} — {ARCH[k]}", transform=axc.transAxes, style="italic",
             fontsize=8.5, color=INK, va="top", ha="left", zorder=8)
    axc.text(0.015, 0.08, f"$r={P[k]['dr_unpol']:.2f}$", transform=axc.transAxes, fontsize=8.3,
             color=BLUE, va="bottom", ha="left", fontweight="bold", zorder=8)
    if gi % 2 == 0:                                               # left column
        axc.set_ylabel("rate (/bp)", fontsize=8.2, color=SUB)
    if gi >= 4:                                                   # bottom row
        axc.tick_params(axis="x", labelsize=8.2, length=2.4, pad=2)
        axc.set_xlabel("position (Mb)", fontsize=8.8, color=SUB, labelpad=2)
    else:
        axc.tick_params(axis="x", labelbottom=False, length=2.0)

fig.text(0.028, 0.765, "b", fontsize=10, fontweight="bold", va="top", ha="left")
fig.text(0.085, 0.765, "real maps from unphased + unpolarized genotypes", fontsize=8.8,
         style="italic", color=SUB,
         va="top", ha="left")
leg_h = [Line2D([0], [0], color=TRUTH, lw=1.4, ls=(0, (4, 2.5)), label="true map"),
         Line2D([0], [0], color=BLUE, lw=1.6, label="DR predicted"),
         Line2D([0], [0], color=BLUE, lw=0.7, ls=(0, (2, 2)), alpha=0.65,
                label="DR 95% bounds")]
fig.legend(handles=leg_h, loc="upper right", bbox_to_anchor=(0.955, 0.768), frameon=False,
           fontsize=7.7, ncol=3, handlelength=1.3, handletextpad=0.5, columnspacing=1.1)

out = os.path.join(FIGDIR, "fig5_unphased.pdf")
fig.savefig(out, dpi=600, facecolor="white")
fig.savefig(os.path.join(FIGDIR, "fig5_unphased.png"), dpi=200, facecolor="white")
print("wrote", out)
