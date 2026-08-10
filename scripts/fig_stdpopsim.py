"""Tree-of-life figure: one frozen fastrho model recovers every stdpopsim genetic map.

Reads results/stdpopsim_<mode>.json (from stdpopsim_maps.py) and the PhyloPic silhouettes
in paper/figures/silhouettes/, draws a per-species map track (true vs fastrho) with the
animal silhouette + genome-wide Pearson r, plus a summary accuracy panel. Runs on the mac.

  python scripts/fig_stdpopsim.py phased
  python scripts/fig_stdpopsim.py unphased
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "paper", "figures")
SILDIR = os.path.join(FIGDIR, "silhouettes")
TRUE_C = ps.C["truth"]
PRED_C = ps.C["fastrho"]
# phylogenetic-ish display order
ORDER = ["human", "orangutan", "baboon", "dog", "fly", "worm", "arabidopsis"]

ps.style()
matplotlib.rcParams.update({
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
    "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#666666", "axes.linewidth": 0.9, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "#b8b8b8", "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
    "legend.fontsize": 9,
})


def add_sil(ax, key, loc=(0.97, 0.95), target_pts=46, alpha=0.92):
    """Place a silhouette PNG (black on transparent) sized to ~target_pts on the long
    edge, so every species renders at a consistent on-page size regardless of source res."""
    try:
        arr = mpimg.imread(os.path.join(SILDIR, key + ".png"))  # RGBA float, native PNG
    except Exception as e:
        print("sil fail", key, e); return
    zoom = target_pts / max(arr.shape[0], arr.shape[1])
    oi = OffsetImage(arr, zoom=zoom, alpha=alpha)
    ab = AnnotationBbox(oi, loc, xycoords="axes fraction", box_alignment=(1, 1),
                        frameon=False, pad=0)
    ax.add_artist(ab)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "phased"
    data = json.load(open(os.path.join(HERE, "paper", "figures",
                                       f"_stdpopsim_{mode}.json")))
    keys = [k for k in ORDER if k in data]

    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 4, hspace=0.52, wspace=0.30)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(len(keys))]

    for ax, key in zip(axes, keys):
        d = data[key]; tr = d["track"]
        c = np.asarray(tr["centers"]); t = np.asarray(tr["truth"]); p = np.asarray(tr["pred"])
        m = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
        ax.plot(c[m], t[m], "-", color=TRUE_C, lw=1.9, label="true map", zorder=4)
        ax.plot(c[m], p[m], "-", color=PRED_C, lw=1.4, alpha=0.85,
                label=("fastrho (DR)" if mode.endswith("_dr") else "fastrho"), zorder=5)
        ax.set_yscale("log")
        ax.set_title(d["common"], fontsize=11, style="italic", pad=4)
        ax.set_xlabel("position (Mb)", labelpad=1)
        ax.margins(x=0.02)
        # headroom for the silhouette
        lo, hi = np.nanmin([t[m].min(), p[m].min()]), np.nanmax([t[m].max(), p[m].max()])
        ax.set_ylim(lo * 0.5, hi * 11)
        add_sil(ax, key, loc=(0.985, 0.97), target_pts=44)
        ax.text(0.03, 0.93, f"$r={d['pearson']:.2f}$", transform=ax.transAxes,
                fontsize=9.5, va="top", ha="left", color=PRED_C,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
    axes[0].set_ylabel("recombination rate (/bp)")
    if len(keys) > 4:
        axes[4].set_ylabel("recombination rate (/bp)")

    # summary accuracy panel in the 8th cell
    axs = fig.add_subplot(gs[1, 3])
    order = sorted(keys, key=lambda k: data[k]["pearson"])
    yy = np.arange(len(order))
    vals = [data[k]["pearson"] for k in order]
    axs.hlines(yy, 0, vals, color="#cfe0ee", lw=2.6, zorder=1)               # lollipop, not bars
    axs.scatter(vals, yy, s=66, color=PRED_C, edgecolor="white", linewidth=0.9, zorder=3)
    for y, v in zip(yy, vals):
        axs.text(v + 0.02, y, f"{v:.2f}", va="center", ha="left", fontsize=8, color=PRED_C)
    axs.set_yticks(yy); axs.set_yticklabels([data[k]["common"] for k in order], fontsize=8.5)
    axs.set_xlim(0, 1.12); axs.set_xlabel("genome-wide Pearson $r$")
    axs.set_title("accuracy across the tree of life", fontsize=10)
    axs.grid(axis="y", alpha=0)

    TAG = {"phased": "",
           "unphased": "  (unphased genotypes — haplotype features)",
           "unphased_gt": "  (unphased genotypes — composite-LD features)",
           "unphased_unpol_gt": "  (unphased + unpolarized — composite-LD, folded)",
           "unphased_unpol_gt_dr": "  (unphased + unpolarized — domain-randomized model)"}
    SUF = {"phased": "", "unphased": "_unphased", "unphased_gt": "_unphased_gt",
           "unphased_unpol_gt": "_unphased_unpol_gt",
           "unphased_unpol_gt_dr": "_unphased_unpol_gt_dr"}
    assert mode in SUF, f"unknown mode {mode}"
    # No bold figure title (Nature style: the caption carries the message).
    # one shared legend
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=2, fontsize=9.5)
    out = os.path.join(FIGDIR, f"fig_stdpopsim{SUF.get(mode, '')}.pdf")
    fig.savefig(out)
    print("wrote", out)
    for k in keys:
        print(f"  {k}: r={data[k]['pearson']:.3f} (log {data[k]['log_pearson']:.3f})")


if __name__ == "__main__":
    main()
