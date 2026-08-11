"""Continuous accuracy-vs-resolution sweep: from large windows (where ReLERNN is fine)
down to fine scale (where fastrho/pyrho pull ahead). Re-bins the stored 25 kb predictions
to a range of window sizes and scores each method, pooled over the headline configs.
Run on sesame (has the configs + matplotlib). -> paper/figures/fig_resolution_sweep.pdf
"""
from __future__ import annotations
import os, glob, json
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11.5, "axes.titlesize": 12.5, "axes.titleweight": "normal",
    "axes.labelsize": 11.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 1.0, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "#b0b0b0", "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.frameon": False,
    "lines.linewidth": 2.6, "lines.markersize": 7,
})
from fastrho.preprocess import mean_rate_between

CAMP = "/home/kkor/fastrho_data/campaign"
CONFIGS = ["const_n20", "const_n40", "real_decode", "real_hapmap"]
BASE = 25_000
GRIDS = [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
METHODS = ["fastrho", "pyrho", "relernn"]
COLORS = {"fastrho": "#1f77b4", "pyrho": "#2ca02c", "relernn": "#d62728"}
LABELS = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}


def block_mean(x, f):
    if f <= 1:
        return x
    n = (len(x) // f) * f
    return x[:n].reshape(-1, f).mean(1)


def truth_windows(npz, L):
    z = np.load(npz, allow_pickle=True)
    edges = np.append(np.arange(0, L, BASE), L)
    return mean_rate_between(z["map_position"], z["map_rate"], edges)


curve = {m: [] for m in METHODS}
for grid in GRIDS:
    f = max(1, grid // BASE)
    pool = {m: ([], []) for m in METHODS}
    for name in CONFIGS:
        cd = os.path.join(CAMP, "configs", name)
        L = json.load(open(os.path.join(cd, "config.json")))["seq_len"]
        truth = {os.path.basename(r)[:-4]: truth_windows(r, L)
                 for r in glob.glob(os.path.join(cd, "region_*.npz"))}
        for m in METHODS:
            p = os.path.join(cd, f"pred_{m}.npz")
            if not os.path.exists(p):
                continue
            pr = np.load(p, allow_pickle=True)
            for rn, tr in truth.items():
                if rn not in pr.files:
                    continue
                k = min(len(pr[rn]), len(tr))
                pool[m][0].append(block_mean(pr[rn][:k], f))
                pool[m][1].append(block_mean(tr[:k], f))
    for m in METHODS:
        if pool[m][0]:
            P = np.concatenate(pool[m][0]); T = np.concatenate(pool[m][1])
            ok = np.isfinite(P) & np.isfinite(T) & (P > 0) & (T > 0)
            curve[m].append(pearsonr(P[ok], T[ok])[0] if ok.sum() > 3 else np.nan)
        else:
            curve[m].append(np.nan)

xs = np.array([g / 1000 for g in GRIDS], float)
fig, ax = plt.subplots(figsize=(7.0, 4.3))
# shade the fine-scale region (where ReLERNN collapses) for emphasis
ax.axvspan(min(xs), 110, color="#ededed", alpha=0.8, zorder=0)
ax.text(60, 0.06, "fine-scale\n(hotspots)", ha="center", va="bottom", fontsize=8.5,
        color="#666666", style="italic")
for m in METHODS:
    y = np.array(curve[m], float)
    ax.plot(xs, y, "-o", label=LABELS[m], color=COLORS[m],
            markeredgecolor="white", markeredgewidth=1.0, zorder=3)
# annotate the fine-scale gap between fastrho and ReLERNN
ax.annotate("", xy=(xs[0], curve["fastrho"][0]), xytext=(xs[0], curve["relernn"][0]),
            arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.3))
ax.text(xs[0] * 0.93, (curve["fastrho"][0] + curve["relernn"][0]) / 2,
        f"+{curve['fastrho'][0]-curve['relernn'][0]:.2f}", ha="right", va="center",
        fontsize=9, color="#444444")
ax.set_xscale("log"); ax.invert_xaxis()         # coarse (left) -> fine (right)
ax.set_xticks(xs); ax.set_xticklabels([f"{int(x)}" for x in xs])
ax.set_xlabel("window size (kb)        coarse  ←        →  fine")
ax.set_ylabel("Pearson $r$ vs. true map")
ax.set_ylim(0, 1.0); ax.legend(loc="center left", title="method")
ax.set_title("Accuracy across resolution: ReLERNN is bound to coarse scale")
fig.tight_layout()
out = "/home/kkor/fastrho/paper/figures/fig_resolution_sweep.pdf"
fig.savefig(out)
print("wrote", out)
for m in METHODS:
    print(m, [None if np.isnan(v) else round(v, 3) for v in curve[m]])
