"""ReLERNN home-turf sweep: where ReLERNN is actually competitive.

Constant-Ne, controlled comparison. Everything below is scored at n=100 haplotypes
(ReLERNN's best regime here -- high SNP density), and we add ReLERNN at n=20 to show
how much sample density lifts it. The x-axis runs from whole-chromosome windows
(coarse, left) to fine scale (right). Even on its home turf -- high density + coarse
windows -- ReLERNN trails fastrho and pyrho at every scale.

Run on sesame. -> paper/figures/fig_hometurf_sweep.pdf
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
BASE = 25_000
GRIDS = [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]

# (label, config, method, color, linestyle, marker)
SERIES = [
    ("fastrho ($n{=}100$)",  "const_n100", "fastrho", "#1f78b4", "-",  "o"),
    ("pyrho ($n{=}100$)",    "const_n100", "pyrho",   "#33a02c", "-",  "s"),
    ("ReLERNN ($n{=}100$)",  "const_n100", "relernn", "#949494", "-",  "o"),
    ("ReLERNN ($n{=}20$)",   "const_n20",  "relernn", "#c4c4c4", "--", "o"),
]


def block_mean(x, f):
    if f <= 1:
        return x
    n = (len(x) // f) * f
    return x[:n].reshape(-1, f).mean(1)


def truth_windows(npz, L):
    z = np.load(npz, allow_pickle=True)
    edges = np.append(np.arange(0, L, BASE), L)
    return mean_rate_between(z["map_position"], z["map_rate"], edges)


def curve_for(config, method):
    cd = os.path.join(CAMP, "configs", config)
    L = json.load(open(os.path.join(cd, "config.json")))["seq_len"]
    truth = {os.path.basename(r)[:-4]: truth_windows(r, L)
             for r in glob.glob(os.path.join(cd, "region_*.npz"))}
    p = os.path.join(cd, f"pred_{method}.npz")
    if not os.path.exists(p):
        return [np.nan] * len(GRIDS)
    pr = np.load(p, allow_pickle=True)
    out = []
    for grid in GRIDS:
        f = max(1, grid // BASE)
        P, T = [], []
        for rn, tr in truth.items():
            if rn not in pr.files:
                continue
            k = min(len(pr[rn]), len(tr))
            P.append(block_mean(pr[rn][:k], f)); T.append(block_mean(tr[:k], f))
        if not P:
            out.append(np.nan); continue
        P = np.concatenate(P); T = np.concatenate(T)
        ok = np.isfinite(P) & np.isfinite(T) & (P > 0) & (T > 0)
        out.append(pearsonr(P[ok], T[ok])[0] if ok.sum() > 3 else np.nan)
    return out


xs = np.array([g / 1000 for g in GRIDS], float)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
# fine-scale = where ReLERNN collapses; coarse end = where it becomes competitive
ax.axvspan(min(xs), 110, color="#ededed", alpha=0.8, zorder=0)
ax.text(45, 0.05, "fine-scale\n(hotspots)", ha="center", va="bottom", fontsize=8.5,
        color="#666666", style="italic")
ax.axvspan(700, max(xs), color="#eaf3ea", alpha=0.9, zorder=0)
ax.text(1150, 0.96, "ReLERNN's regime", ha="center", va="top", fontsize=8.5,
        color="#3a7a3a", style="italic")

curves = {}
for lab, cfg, m, col, ls, mk in SERIES:
    y = np.array(curve_for(cfg, m), float)
    curves[lab] = y
    ax.plot(xs, y, ls, marker=mk, label=lab, color=col,
            markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, alpha=0.95 if ls == "-" else 0.9)

n100 = curves["ReLERNN ($n{=}100$)"]; n20 = curves["ReLERNN ($n{=}20$)"]
# ReLERNN only reaches parity-ish at the coarsest (whole-chromosome 2 Mb) scale (index -1)
if np.isfinite(n100[-1]):
    ax.annotate(f"competitive only here:\n$r\\approx{n100[-1]:.2f}$ at 2 Mb\n(still $<$ fastrho/pyrho)",
                xy=(xs[-1], n100[-1]), xytext=(900, 0.79), fontsize=8.5, color="#666666",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=1.1))
# the two ReLERNN densities overlap -> sample density is not the lever
mid = 3  # ~250 kb index
ax.annotate("$n{=}20$ and $n{=}100$ overlap:\nsample density doesn't help\n(SNPs/window capped)",
            xy=(xs[mid], (n100[mid] + n20[mid]) / 2), xytext=(250, 0.27),
            fontsize=8.5, color="#666666", ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.0))

ax.set_xscale("log"); ax.invert_xaxis()
ax.set_xticks(xs); ax.set_xticklabels([f"{int(x)}" for x in xs])
ax.set_xlabel("window size (kb)        coarse  ←        →  fine")
ax.set_ylabel("Pearson $r$ vs. true map")
ax.set_ylim(0, 1.0)
ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.01), title="method (constant $N_e$)")
ax.set_title("Where ReLERNN works: only the coarsest, whole-chromosome scale")
fig.tight_layout()
out = "/home/kkor/fastrho/paper/figures/fig_hometurf_sweep.pdf"
fig.savefig(out)
print("wrote", out)
for lab in curves:
    print(lab, [None if np.isnan(v) else round(v, 3) for v in curves[lab]])
