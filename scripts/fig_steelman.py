"""Extended Data: a per-SNP GRU on raw genotypes (ReLERNN-seq2seq) is not enough.

Steelman check -- we give ReLERNN's recurrent backbone a per-SNP seq2seq head (so it is no longer
window-bound) and it still falls far short of fastrho at fine scale, across every benchmark
configuration. Rendered as a dumbbell/dot plot (not grouped bars): each row is one configuration,
the three dots are the methods' 25 kb Pearson r, and the gap between fastrho and the two ReLERNN
variants is the point.

Reads paper/results_snapshot/summary.json.  Run: python scripts/fig_steelman.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMM = os.path.join(_HERE, "paper", "results_snapshot", "summary.json")
OUT = os.path.join(_HERE, "paper", "figures")
CFG = [("const_n20", "constant  $n{=}20$"), ("const_n40", "constant  $n{=}40$"),
       ("real_decode", "deCODE"), ("real_hapmap", "HapMap II")]
METHODS = [("fastrho", "fastrho", ps.C["fastrho"]),
           ("gruseq2seq", "ReLERNN-seq2seq", ps.C["gru"]),
           ("relernn", "ReLERNN", ps.C["relernn"])]
ps.style()


def main():
    s = json.load(open(SUMM))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ys = np.arange(len(CFG))[::-1]                       # first config at top
    for y, (cfg, _) in zip(ys, CFG):
        sc = s[cfg]["scales"]["25kb"]
        vals = [float(sc[m]["pearson"]) for m, _, _ in METHODS]
        ax.plot([min(vals), max(vals)], [y, y], color="#cfcfcf", lw=2.4, zorder=1,
                solid_capstyle="round")
        for (m, _, col), v in zip(METHODS, vals):
            ax.scatter(v, y, s=95, color=col, edgecolor="white", linewidth=1.1, zorder=3)
        # label fastrho value (right of its dot); positions + legend convey the rest
        ax.annotate(f"{vals[0]:.2f}", (vals[0], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.4, color=METHODS[0][2], fontweight="medium")
    ax.set_yticks(ys); ax.set_yticklabels([lab for _, lab in CFG], fontsize=9.4)
    ax.set_xlim(0, 1.02); ax.set_ylim(-0.6, len(CFG) - 0.4)
    ax.set_xlabel("Pearson $r$ vs. true map  (25 kb)")
    ax.grid(axis="x", color="#b8b8b8", alpha=0.28, lw=0.6); ax.grid(axis="y", visible=False)
    ax.spines["left"].set_color("#888"); ax.spines["bottom"].set_color("#888")
    ax.set_title("A per-SNP GRU on raw genotypes (ReLERNN-seq2seq) still falls far short of fastrho",
                 loc="left", fontsize=10)
    handles = [Line2D([0], [0], marker="o", ls="none", mfc=c, mec="white", ms=8, label=lab)
               for _, lab, c in METHODS]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.60, 0.60), fontsize=8.4,
              handletextpad=0.3, borderpad=0.6, framealpha=0.9)
    out = os.path.join(OUT, "fig_steelman.pdf")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
