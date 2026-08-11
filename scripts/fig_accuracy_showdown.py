"""Main figure 2 (merged): fastrho vs ReLERNN, shown clearly.

Combines the quantitative head-to-head (old Fig 2) with the visual resolution zoom (old Fig 12)
into one figure whose single message is that fastrho resolves fine-scale structure ReLERNN cannot:
  (a) ReLERNN's correlation collapses as the scale gets finer, while fastrho and pyrho hold.
  (b) the gap is systematic across every benchmark dataset (25 kb fine scale).
  (c) the smoking gun -- a real deCODE hotspot: fastrho tracks it per SNP interval, ReLERNN
      averages it into one flat ~80 kb window.

Pure plotting on the mac (python3.13): reads paper/figdata/ only.
  python3.13 scripts/fig_accuracy_showdown.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as ps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, "paper", "figdata")
OUT = os.path.join(ROOT, "paper", "figures")
C = {"fastrho": ps.C["fastrho"], "pyrho": ps.C["pyrho"], "relernn": ps.C["relernn"],
     "truth": ps.C["truth"], "band": ps.C["fastrho_l"]}
LAB = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}
NICE = {"const_n20": "constant $n{=}20$", "const_n40": "constant $n{=}40$",
        "real_hapmap": "HapMap II", "real_decode": "deCODE"}
ps.style()
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.2,
                     "axes.grid": True, "grid.alpha": 0.18})


def step_path(edges, vals):
    return np.repeat(edges, 2)[1:-1], np.repeat(vals, 2)


def panel(ax, s, x=-0.085, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")


def main():
    z = np.load(os.path.join(FD, "relernn_showdown.npz"), allow_pickle=True)
    m = json.loads(str(z["meta"]))
    dots = m["dots"]; xs = np.array(m["grids_kb"]); curve = m["curve"]
    mpos = z["mpos"]; mrate = z["mrate"]; fc = z["fc"]; fr = z["fr"]
    flo = z["flo"]; fhi = z["fhi"]; rel_e = z["rel_edges"]; rel_r = z["rel_rates"]
    hc = m["hc"]; L = m["L"]; methods = ["fastrho", "pyrho", "relernn"]

    fig = plt.figure(figsize=(12.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.46, wspace=0.23,
                          top=0.865, bottom=0.08, left=0.07, right=0.975)

    # ---- (a) resolution-degradation curve ----
    a = fig.add_subplot(gs[0, 0])
    a.axvspan(min(xs), 110, color="#f2f2f2", zorder=0)
    a.text(46, 0.06, "fine scale\n(hotspots)", ha="center", va="bottom", fontsize=8,
           color="#888", style="italic")
    zo = {"fastrho": 6, "pyrho": 5, "relernn": 4}
    for mm in methods:
        a.plot(xs, curve[mm], "-o", color=C[mm], markeredgecolor="white", markeredgewidth=1.0,
               lw=2.4, ms=6.5, label=LAB[mm], zorder=zo[mm])
    g = curve["fastrho"][0] - curve["relernn"][0]
    a.annotate("", xy=(xs[0], curve["fastrho"][0]), xytext=(xs[0], curve["relernn"][0]),
               arrowprops=dict(arrowstyle="<->", color="#666", lw=1.2))
    a.text(xs[0] * 1.95, (curve["fastrho"][0] + curve["relernn"][0]) / 2,
           "$\\Delta r=%.2f$\nat 25 kb" % g, ha="center", va="center", fontsize=9, color="#444")
    a.text(1100, 0.75, "ReLERNN:\n1 rate / window", ha="center", va="center",
           fontsize=8.4, color="#6f6f6f")
    a.set_xscale("log"); a.invert_xaxis()
    a.set_xticks(xs); a.set_xticklabels([f"{int(v)}" for v in xs], fontsize=8.3)
    a.set_xlabel("$\\leftarrow$ coarser     window size (kb)     finer $\\rightarrow$")
    a.set_ylabel("Pearson $r$ vs. true map"); a.set_ylim(0, 1.0)
    a.set_title("ReLERNN's accuracy collapses at fine scale", loc="left")
    panel(a, "a")

    # ---- (b) the meaningful head-to-head: fastrho vs pyrho at 25 kb, edge grows off-equilibrium ----
    b = fig.add_subplot(gs[0, 1])
    summ = json.load(open(os.path.join(ROOT, "paper", "results_snapshot", "summary.json")))

    def acc(cfg, mm):
        try:
            return float(summ[cfg]["scales"]["25kb"][mm]["pearson"])
        except (KeyError, TypeError):
            return None
    NICE2 = {**NICE, "bottleneck_n20": "bottleneck $n{=}20$", "expansion_n20": "expansion $n{=}20$"}
    # ordered by fastrho-minus-pyrho margin (largest first): the edge is biggest where the
    # composite-likelihood is starved -- non-equilibrium demography and small samples.
    cfgs = ["bottleneck_n20", "expansion_n20", "real_decode", "const_n20", "real_hapmap",
            "const_n40", "const_n100"]
    NICE2["const_n100"] = "constant $n{=}100$"
    yy = np.arange(len(cfgs))[::-1]
    for cfg, yi in zip(cfgs, yy):
        f, p = acc(cfg, "fastrho"), acc(cfg, "pyrho")
        b.plot([p, f], [yi, yi], color="#d7d7d7", lw=2.6, zorder=1, solid_capstyle="round")
        b.scatter(p, yi, s=84, color=C["pyrho"], edgecolor="white", linewidth=1.1, zorder=4)
        b.scatter(f, yi, s=92, color=C["fastrho"], edgecolor="white", linewidth=1.1, zorder=5)
        m = f - p
        b.annotate("$%+.2f$" % m, (max(f, p), yi), xytext=(8, 0), textcoords="offset points",
                   ha="left", va="center", fontsize=8.2, fontweight="medium",
                   color=(C["fastrho"] if m >= 0 else "#888"))
    b.axhspan(len(cfgs) - 2.5, len(cfgs) - 0.5, color="#eef2f6", zorder=0)
    b.text(0.375, len(cfgs) - 0.58, "non-equilibrium & small $n$:\nfastrho's edge is largest",
           fontsize=7.4, color="#5b7ea6", va="top", style="italic")
    b.set_yticks(yy); b.set_yticklabels([NICE2[c] for c in cfgs], fontsize=8.6)
    b.set_xlim(0.35, 1.0); b.set_xticks(np.arange(0.4, 1.01, 0.2))
    b.set_ylim(-0.6, len(cfgs) - 0.4)
    b.grid(axis="y", visible=False); b.grid(axis="x", alpha=0.25)
    b.set_xlabel("Pearson $r$ vs. true map  (25 kb)")
    b.set_title(r"fastrho matches or beats pyrho, edge largest where likelihood info is scarce",
                loc="left")
    panel(b, "b")

    # ---- (c) On ReLERNN's own showcase benchmark (Comeron 2L), fastrho resolves the true map per
    #      interval while ReLERNN averages every hotspot into a flat staircase -- the resolution gap,
    #      direct. fastrho r = 0.97 / 0.92 at 100 / 25 kb on this region (repro_showdown metrics). ----
    floor = 1.2e-9
    rp = np.load(os.path.join(FD, "repro_showdown.npz"), allow_pickle=True)
    cx = np.asarray(rp["centers_hi_mb"], float)
    ct = np.clip(np.asarray(rp["truth_hi"], float), floor, None)
    cf = np.clip(np.asarray(rp["fastrho_hi"], float), floor, None)
    cp = np.clip(np.asarray(rp["pyrho_hi"], float), floor, None)
    cr = np.clip(np.asarray(rp["relernn_hi"], float), floor, None)
    zlo, zhi, ylo, yhi = 16.4, 19.2, 1.0e-9, 1.6e-7   # a representative multi-hotspot stretch, not a worst case
    axr = fig.add_subplot(gs[1, :])
    axr.fill_between(cx, floor, ct, color="#e6e9ec", lw=0, zorder=0)                        # true-map fill
    axr.plot(cx, cr, color=C["relernn"], lw=2.8, zorder=2, solid_capstyle="butt")          # ReLERNN staircase
    axr.plot(cx, ct, color="k", lw=1.8, zorder=3)                                          # true map (bold)
    axr.plot(cx, cp, color=C["pyrho"], lw=1.7, zorder=4)                                    # pyrho (gold standard)
    axr.plot(cx, cf, color=C["fastrho"], lw=2.1, zorder=5)                                  # fastrho (on the truth)
    axr.set_yscale("log"); axr.set_xlim(zlo, zhi); axr.set_ylim(ylo, yhi)
    axr.set_yticks([1e-8]); axr.set_yticklabels(["$10^{-8}$"], fontsize=8)
    axr.grid(axis="y", alpha=0.15)
    axr.set_xlabel("chromosome position (Mb)"); axr.set_ylabel("recombination rate (bp$^{-1}$)")
    axr.set_title("On ReLERNN's own benchmark (Comeron 2L): fastrho (blue) and pyrho (green) resolve the "
                  "true map (black) hotspot by hotspot; ReLERNN (grey), one rate per ~100 kb window, is coarse",
                  loc="left")
    for nm, cc, yy in [("fastrho", C["fastrho"], 0.96), ("pyrho", C["pyrho"], 0.86),
                       ("true map", "k", 0.76), ("ReLERNN", C["relernn"], 0.66)]:
        axr.text(0.992, yy, nm, transform=axr.transAxes, color=cc, fontweight="bold",
                 va="top", ha="right", fontsize=9)
    panel(axr, "c", x=-0.065, y=1.10)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#e9edf0", edgecolor="#c3c8ce", label="true map (shaded)"),
               Line2D([0], [0], color=C["fastrho"], lw=2.4, marker="o", mfc=C["fastrho"],
                      mec="white", ms=7, label="fastrho"),
               Line2D([0], [0], color=C["pyrho"], lw=2.4, marker="o", mfc=C["pyrho"],
                      mec="white", ms=7, label="pyrho"),
               Line2D([0], [0], color=C["relernn"], lw=2.8, marker="o", mfc=C["relernn"],
                      mec="white", ms=7, label="ReLERNN")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=5,
               frameon=False, fontsize=10.5, columnspacing=1.9, handletextpad=0.6, handlelength=1.8)
    out = os.path.join(OUT, "fig2_accuracy.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
