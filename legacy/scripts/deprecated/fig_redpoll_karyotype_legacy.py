"""Arrangement-specific redpoll supergene analysis.

Shows that the deep pooled LD trough across the chromosome-1 inversion is
substantially weakened when maps are inferred separately within the two
homokaryotype arrangements.  Random mixed-karyotype subsets control for the
smaller sample sizes.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paper_style as ps
from fastrho.preprocess import mean_rate_between


ps.style()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "paper", "figdata")
fg = np.load(os.path.join(DATA, "fieldguide_redpoll.npz"), allow_pickle=True)
km = np.load(os.path.join(DATA, "redpoll_karyotype_maps.npz"), allow_pickle=True)
null = json.load(open(os.path.join(DATA, "redpoll_karyotype_null.json")))
ld = json.load(open(os.path.join(DATA, "redpoll_karyotype_ld.json")))

BLUE = ps.C["fastrho"]
ORANGE = ps.CB[5]
GREY = ps.C["relernn"]
PALE = "#d7d7d7"
INV = "#f6d6b6"


def pooled_map():
    bp = np.r_[fg["pos_left"][0], fg["pos_right"]]
    return mean_rate_between(bp, fg["rho_per_bp"], km["edges"])


def ratio(x, inside, flank):
    return float(np.nanmedian(x[inside]) / np.nanmedian(x[flank]))


def smooth(x, n=4):
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")


centers = km["centers"] / 1e6
inv0 = float(km["inv_start"]) / 1e6
inv1 = float(km["inv_end"]) / 1e6
inside = (centers >= inv0) & (centers < inv1)
flank = ~inside
pooled = pooled_map()
a = km["arrangement_A_rate"]
b = km["arrangement_B_rate"]

ratios = {
    "Pooled": ratio(pooled, inside, flank),
    "Arrangement A": ratio(a, inside, flank),
    "Arrangement B": ratio(b, inside, flank),
}
null37 = [r["inside_flank_ratio"] for r in null["records"] if r["size"] == 37]
null28 = [r["inside_flank_ratio"] for r in null["records"] if r["size"] == 28]

fig = plt.figure(figsize=(14.2, 4.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.58, 0.98, 1.12], wspace=0.43,
                      left=0.06, right=0.985, bottom=0.17, top=0.86)

# (a) pooled and arrangement-specific maps ------------------------------------------------
ax = fig.add_subplot(gs[0])
ax.axvspan(inv0, inv1, color=INV, alpha=0.72, zorder=0)
ax.plot(centers, smooth(pooled), color=GREY, lw=2.1, label="pooled panel (72 birds)")
ax.plot(centers, smooth(a), color=BLUE, lw=2.1, label="arrangement A homozygotes ($n=37$)")
ax.plot(centers, smooth(b), color=ORANGE, lw=2.1, label="arrangement B homozygotes ($n=28$)")
ax.set_yscale("log")
ax.set_xlabel("position on redpoll chromosome 1 (Mb)")
ax.set_ylabel(r"population-scaled rate, $\hat\rho$ (smoothed 2 Mb)")
ax.set_title("The pooled supergene cold block weakens within arrangements", loc="left")
ax.text((inv0 + inv1) / 2, ax.get_ylim()[1] / 1.20, "55-Mb supergene inversion",
        ha="center", va="top", fontsize=8.4, color="#8a5a2b")
ax.legend(loc="lower right", fontsize=8.2, handlelength=1.8)
ps.panel(ax, "a", x=-0.10, y=1.11)

# (b) ratios plus matched-n null -----------------------------------------------------------
ax = fig.add_subplot(gs[1])
xpos = np.array([0.0, 1.0, 2.0])
vals = [ratios["Pooled"], ratios["Arrangement A"], ratios["Arrangement B"]]
colors = [GREY, BLUE, ORANGE]
ax.bar(xpos, vals, width=0.58, color=colors, edgecolor="white", zorder=2)
rng = np.random.default_rng(17)
for x, vv in [(1.0, null37), (2.0, null28)]:
    jitter = rng.uniform(-0.18, 0.18, len(vv))
    ax.scatter(x + jitter, vv, s=30, facecolor="white", edgecolor="#555", lw=0.8,
               zorder=4, label="mixed-karyotype size null" if x == 1.0 else None)
for x, v, c in zip(xpos, vals, colors):
    ax.text(x, v + 0.025, f"{v:.2f}", ha="center", va="bottom", color=c,
            fontsize=9.2, fontweight="bold")
ax.axhline(1, color="#aaa", lw=1, ls=(0, (4, 3)))
ax.set_ylim(0, 1.10)
ax.set_xticks(xpos)
ax.set_xticklabels(["pooled", "A homozygotes\n($n=37$)", "B homozygotes\n($n=28$)"])
ax.set_ylabel("median rate inside / flanks")
ax.set_title("Arrangement mixture creates most of the cold block", loc="left", fontsize=10.8)
ax.legend(loc="upper left", fontsize=7.4, handletextpad=0.2)
ps.panel(ax, "b", x=-0.14, y=1.11)

# (c) raw LD-decay confirmation, independent of the inferred maps -------------------------
ax = fig.add_subplot(gs[2])
for key, label, color in [
    ("pooled", "pooled", GREY),
    ("arrangement_A", "arrangement A", BLUE),
    ("arrangement_B", "arrangement B", ORANGE),
]:
    d = ld["groups"][key]
    x = np.array([r["distance_mid"] for r in d["inside"]]) / 1e3
    y_in = np.array([r["mean_r2_corrected"] for r in d["inside"]])
    y_out = np.array([r["mean_r2_corrected"] for r in d["flanks"]])
    ax.plot(x, y_in, color=color, lw=2.2, label=label)
    ax.plot(x, y_out, color=color, lw=1.35, ls=(0, (3, 2)), alpha=0.8)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("physical separation between SNPs (kb)")
ax.set_ylabel(r"mean dosage $r^2$ (finite-sample corrected)")
ax.set_title("Raw LD decays within each arrangement", loc="left")
ax.text(0.97, 0.96, "solid: inside inversion\ndashed: collinear flanks",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color="#555")
ax.text(0.04, 0.08, "at 250–500 kb:\npooled inside LD is 15–30×\nhigher than within arrangements",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8.1, color="#555")
ax.legend(loc="center right", fontsize=8.0)
ps.panel(ax, "c", x=-0.12, y=1.11)

ps.save(fig, "fig_redpoll_karyotype", formats=("pdf", "png"))
