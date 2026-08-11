"""Extended Data: why ReLERNN trails at fine scale (consolidates three former ED figures).

One figure makes the whole ReLERNN-comparison point, from three angles:
(a) Accuracy vs scoring window: fastrho and pyrho are scale-robust; ReLERNN's correlation rises
    only toward the coarse (whole-chromosome) scale and collapses at fine scale.
(b) ReLERNN's own hotspot diagnostic (their SI Fig. S13): a single hotspot of increasing length in
    a 250 kb region. fastrho (per SNP interval) recovers a growing fraction of the true 50x
    enrichment; ReLERNN (one rate per ~100 kb window) never leaves the no-detection floor.
(c) Steelman: even a per-SNP GRU on raw genotypes (ReLERNN-seq2seq, no longer window-bound) stays
    far below fastrho at 25 kb on every benchmark -- so the gap is fastrho's LD-aware features and
    state-space backbone, not output granularity.

Reads relernn_showdown.npz, hotspot_length.json, results_snapshot/summary.json.
Run: python scripts/fig_relernn_scale.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(_HERE, "paper", "figdata")
RS = os.path.join(_HERE, "paper", "results_snapshot")
OUT = os.path.join(_HERE, "paper", "figures")
ps.style()
BLUE, GREEN, GREY, PUR = ps.C["fastrho"], ps.C["pyrho"], ps.C["relernn"], ps.C["gru"]


def panel_sweep(a):
    d = np.load(os.path.join(FD, "relernn_showdown.npz"), allow_pickle=True)
    m = json.loads(str(d["meta"]))
    g = np.array(m["grids_kb"], float)
    a.axvspan(g.min(), 100, color="#eef3f7", zorder=0)                # the fine-scale regime
    for key, col, lab in [("fastrho", BLUE, "fastrho"), ("pyrho", GREEN, "pyrho"),
                          ("relernn", GREY, "ReLERNN")]:
        a.plot(g, m["curve"][key], "-o", color=col, ms=5, lw=2.4, label=lab)
    a.set_xscale("log"); a.invert_xaxis(); a.set_ylim(0, 1.0)
    a.set_xlabel("scoring window (kb)      coarse $\\leftarrow$   $\\rightarrow$ fine")
    a.set_ylabel("Pearson $r$ vs. true map")
    a.legend(fontsize=8.6, loc="lower left")
    a.text(60, 0.30, "fine-scale\nregime", fontsize=8.0, color="#6b8299", ha="center")
    a.set_title("(a) scale-robust for fastrho/pyrho, coarse-only for ReLERNN", loc="left", fontsize=9.6)
    ps.panel(a, "a", x=-0.20)


def panel_hotspot(b):
    hs = json.load(open(os.path.join(FD, "hotspot_length.json")))
    lens = sorted(int(k) for k in hs)
    fx = np.array([hs[str(L)]["fastrho"]["mean"] for L in lens])
    fse = np.array([hs[str(L)]["fastrho"]["se"] for L in lens])
    rx = np.array([hs[str(L)]["relernn"]["mean"] for L in lens])
    rse = np.array([hs[str(L)]["relernn"]["se"] for L in lens])
    b.axhline(50, color="#999", ls=":", lw=1.2); b.text(9.9, 50, "true (50$\\times$)",
                                                        fontsize=8, color="#777", va="top", ha="right")
    b.axhline(1, color="#bbb", ls="--", lw=1.0); b.text(9.9, 2.6, "no detection (1$\\times$)",
                                                        fontsize=8, color="#999", ha="right")
    b.errorbar(lens, fx, yerr=fse, color=BLUE, lw=2.4, marker="o", ms=5, capsize=2.5,
               label="fastrho (per SNP interval)")
    b.errorbar(lens, rx, yerr=rse, color=GREY, lw=2.4, marker="o", ms=5, capsize=2.5,
               label="ReLERNN (1 rate / window)")
    b.set_ylim(0, 53); b.set_xlabel("hotspot length (kb)")
    b.set_ylabel("predicted enrichment ($\\times$ background)")
    b.legend(fontsize=8.4, loc="center left", bbox_to_anchor=(0.02, 0.62))
    b.set_title("(b) short-hotspot detectability (ReLERNN SI Fig. S13)", loc="left", fontsize=9.6)
    ps.panel(b, "b", x=-0.20)


def panel_steelman(c):
    s = json.load(open(os.path.join(RS, "summary.json")))
    cfg = [("const_n20", "constant $n{=}20$"), ("const_n40", "constant $n{=}40$"),
           ("real_decode", "deCODE"), ("real_hapmap", "HapMap II")]
    methods = [
        ("fastrho", BLUE, "o"),
        ("gruseq2seq", PUR, "s"),
        ("relernn", GREY, "^"),
    ]
    ys = np.arange(len(cfg))[::-1]
    for y, (k, _) in zip(ys, cfg):
        sc = s[k]["scales"]["25kb"]
        vals = [float(sc[m]["pearson"]) for m, _, _ in methods]
        c.plot([min(vals), max(vals)], [y, y], color="#cfcfcf", lw=2.4, zorder=1, solid_capstyle="round")
        for (m, col, marker), v in zip(methods, vals):
            c.scatter(
                v,
                y,
                s=80,
                color=col,
                marker=marker,
                edgecolor="white",
                linewidth=1.0,
                zorder=3,
            )
        c.annotate(f"{vals[0]:.2f}", (vals[0], y), xytext=(7, 0), textcoords="offset points",
                   va="center", fontsize=8.0, color=BLUE, fontweight="medium")
    c.set_yticks(ys); c.set_yticklabels([lab for _, lab in cfg], fontsize=8.6)
    c.set_xlim(0, 1.02); c.set_ylim(-0.6, len(cfg) - 0.4)
    c.set_xlabel("Pearson $r$ (25 kb)")
    c.grid(False)
    handles = [Line2D([0], [0], marker=marker, ls="none", mfc=col, mec="white", ms=7,
                      label={"fastrho": "fastrho", "gruseq2seq": "ReLERNN-seq2seq", "relernn": "ReLERNN"}[m])
               for m, col, marker in methods]
    c.legend(handles=handles, loc="center", bbox_to_anchor=(0.62, 0.55), fontsize=8.0,
             handletextpad=0.3, borderpad=0.5, framealpha=0.9)
    c.set_title("(c) matched baseline: a per-SNP GRU on raw genotypes is not enough", loc="left", fontsize=9.6)
    ps.panel(c, "c", x=-0.22)


def main():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.3))
    panel_sweep(a); panel_hotspot(b); panel_steelman(c)
    out = os.path.join(OUT, "fig_relernn_scale.pdf")
    fig.tight_layout(w_pad=2.2); fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
