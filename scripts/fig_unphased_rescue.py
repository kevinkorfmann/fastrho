"""Appendix figure: composite-LD features rescue unphased accuracy with no retraining.

Three-way per-species comparison of genome-wide pooled Pearson r:
  phased                      -- haplotype features, phased data (reference)
  unphased, haplotype feats   -- the naive path collapses
  unphased, composite-LD      -- the same frozen model with phase-invariant tokens
Reads the three results/stdpopsim_*.json (copied into paper/figures as _stdpopsim_*.json).
Run on sesame or the mac.
"""
from __future__ import annotations
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "paper", "figures")
ORDER = ["human", "orangutan", "baboon", "dog", "fly", "worm", "arabidopsis"]
# condition palette (fastrho-internal; NOT the method palette): reference / dead / 2 rescues
SERIES = [
    ("phased", "_stdpopsim_phased.json", "phased haplotypes (reference)", "#0173B2"),
    ("broken", "_stdpopsim_unphased.json", "unphased — haplotype features (naive)", "#9e9e9e"),
    ("rescue", "_stdpopsim_unphased_gt.json", "unphased — composite-LD features", "#17a2b8"),
    ("rescue2", "_stdpopsim_unphased_unpol_gt.json",
     "unphased + unpolarized — composite-LD, folded", "#0b5563"),
]

ps.style()
matplotlib.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "normal",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 1.0, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "#b0b0b0", "grid.alpha": 0.30, "grid.linewidth": 0.6,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.frameon": False,
})

data = {key: json.load(open(os.path.join(FIGDIR, fn))) for key, fn, _, _ in SERIES}
keys = [k for k in ORDER if k in data["phased"]]
labels = [data["phased"][k]["common"] for k in keys]
x = np.arange(len(keys)); nser = len(SERIES); w = 0.2

fig, ax = plt.subplots(figsize=(11.0, 4.3))
for i, (key, _, lab, col) in enumerate(SERIES):
    vals = [data[key].get(k, {}).get("pearson", np.nan) for k in keys]
    bars = ax.bar(x + (i - (nser - 1) / 2) * w, vals, w, label=lab, color=col)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=6.6, color="#444444", rotation=90,
                        xytext=(0, 2), textcoords="offset points")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, 1.18); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("genome-wide pooled Pearson $r$")
ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          columnspacing=1.6, handletextpad=0.5, borderaxespad=0.0)
ax.set_title("Composite-LD features rescue unphased (and unpolarized) accuracy — no retraining",
             pad=42)
fig.tight_layout()
out = os.path.join(FIGDIR, "fig_unphased_rescue.pdf")
fig.savefig(out)
print("wrote", out)
for k in keys:
    print(f"  {data['phased'][k]['common']:14s} phased={data['phased'][k]['pearson']:.3f}  "
          f"broken={data['broken'].get(k,{}).get('pearson',float('nan')):.3f}  "
          f"rescue={data['rescue'].get(k,{}).get('pearson',float('nan')):.3f}")
