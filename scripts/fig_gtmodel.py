"""Appendix figure: a dedicated dosage model closes the unphased+unpolarized gap.

Per-species genome-wide pooled Pearson r on unphased + unpolarized genotypes:
  phased reference            -- the ceiling (phased haplotypes, original model)
  base model + folded feats   -- the frozen base model fed composite-LD/folded tokens (no retrain)
  dedicated GT model          -- a model trained on folded composite-LD tokens
Reads paper/figures/_stdpopsim_*.json. Run on sesame or the mac.
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
# all three are fastrho variants -> a blue gradient (reference / no-retrain / retrained)
SERIES = [
    ("phased", "_stdpopsim_phased.json", "phased reference (ceiling)", "#0173B2"),
    ("e1", "_stdpopsim_unphased_unpol_gt.json",
     "unphased+unpolarized: base model + folded features (no retrain)", "#a6cee3"),
    ("e2", "_stdpopsim_unphased_unpol_gt_gtmodel.json",
     "unphased+unpolarized: dedicated GT model (retrained)", "#08519c"),
]

ps.style()
matplotlib.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "normal",
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#555555", "axes.linewidth": 1.0, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "#b0b0b0", "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.frameon": False,
})

data = {k: json.load(open(os.path.join(FIGDIR, fn))) for k, fn, _, _ in SERIES}
keys = [k for k in ORDER if k in data["phased"]]
labels = [data["phased"][k]["common"] for k in keys]
x = np.arange(len(keys)); nser = len(SERIES); w = 0.26

fig, ax = plt.subplots(figsize=(9.6, 4.5))
for i, (key, _, lab, col) in enumerate(SERIES):
    vals = [data[key].get(k, {}).get("pearson", np.nan) for k in keys]
    bars = ax.bar(x + (i - (nser - 1) / 2) * w, vals, w, label=lab, color=col, zorder=3)
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=6.8, color="#444444", rotation=90,
                        xytext=(0, 2), textcoords="offset points")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0, 1.16); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("genome-wide pooled Pearson $r$")
ax.legend(ncol=1, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          handletextpad=0.5, borderaxespad=0.0)
ax.set_title("A dedicated dosage model closes the unphased + unpolarized gap", pad=74)
fig.tight_layout()
out = os.path.join(FIGDIR, "fig_gtmodel.pdf")
fig.savefig(out)
print("wrote", out)
for k in keys:
    print(f"  {data['phased'][k]['common']:13s} phased={data['phased'][k]['pearson']:.3f}  "
          f"E1={data['e1'].get(k,{}).get('pearson',float('nan')):.3f}  "
          f"E2={data['e2'].get(k,{}).get('pearson',float('nan')):.3f}")
