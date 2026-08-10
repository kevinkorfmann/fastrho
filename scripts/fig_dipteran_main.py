"""Extended Data (Fig. dipteran): absolute-rate calibration holds in the extreme-Ne regime.

At Ne~1e6 (dipterans) LD saturates within ~1 kb, so the *absolute* background recombination
level is only weakly identified -- the regime that makes LD-based absolute-rate inference hard.
The high-Ne-trained fastrho model (campaign_hidip) recovers the landscape AND its absolute scale
directly, in one forward pass, with NO anchor: bias ratio ~0.90 on Drosophila and ~0.77 on the
most extreme synthetic Anopheles maps, up from ~0.50-0.63 for the earlier model. This converts a
former limitation (uniform absolute-rate shrinkage, previously corrected post-hoc by a genome-mean
anchor) into a calibration strength.

  (a) one extreme-Ne (Ne=1e6) Drosophila region recovered per SNP interval: truth (2-level step),
      the earlier model (background shrunk to ~0.45x), and the high-Ne-fixed model -- which lands
      on the truth with no anchor. Faint = per-SNP; bold = 2 kb binned mean (readability only).
  (b) the absolute offset is gone: distribution of predicted/true rate for that same region.
  (c) systematic across the regime: landscape Pearson stays high AND the absolute bias ratio moves
      toward 1 for both Drosophila and Anopheles (earlier model in grey for contrast).

House style: paper_style fonts + ps.panel letters + locked palette (no ad-hoc hex, no font override).
All numbers are read from paper/figdata/ so the figure and caption cannot drift apart.
Pure plotting on the mac (python3.13):  python3.13 scripts/fig_dipteran_main.py
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

BLACK, BLUE, GREY = ps.C["truth"], ps.C["fastrho"], ps.C["relernn"]   # locked palette
STEM = "#d9d9d9"
ps.style()
plt.rcParams.update({"grid.alpha": 0.16})
TITLE_FS = 9.6


def step_xy(mp, mr):
    return np.repeat(mp, 2)[1:-1], np.repeat(mr, 2)


def binned(x, y, lo, hi, w=2.0):
    """2 kb binned mean -- a readability overlay on the per-SNP track, not a new estimate."""
    edges = np.arange(lo, hi + w, w)
    idx = np.digitize(x, edges)
    xc, ym = [], []
    for i in range(1, len(edges)):
        m = idx == i
        if m.any():
            xc.append(0.5 * (edges[i - 1] + edges[i])); ym.append(float(np.mean(y[m])))
    return np.array(xc), np.array(ym)


def main():
    z = np.load(os.path.join(FD, "dipteran_bias.npz"), allow_pickle=True)
    st = json.loads(str(z["stats"]))
    mid = z["mid"] / 1e3                        # kb
    rhat = z["rhat"]; rhat_old = z["rhat_old"]; rtrue = z["rtrue"]
    mp = z["mp"] / 1e3; mr = z["mr"]
    sc = 1e8                                    # plot in 1e-8 c/bp

    fig = plt.figure(figsize=(12.8, 3.95))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.02, 1.18], wspace=0.36,
                          left=0.055, right=0.99, bottom=0.16, top=0.80)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- (a) extreme-Ne recovery: truth, earlier model (shrunk), fixed model (on truth) ----
    tx, ty = step_xy(mp, mr * sc)
    lo, hi = float(mid.min()), float(mid.max())
    a.fill_between(tx, 0, ty, color=BLACK, alpha=0.06, lw=0, zorder=1)
    a.plot(tx, ty, color=BLACK, lw=1.6, label="true map", zorder=5)
    a.plot(mid, rhat_old * sc, color=GREY, lw=0.5, alpha=0.28, zorder=2)   # per-SNP (faint)
    a.plot(mid, rhat * sc, color=BLUE, lw=0.5, alpha=0.22, zorder=3)
    xo, yo = binned(mid, rhat_old * sc, lo, hi); xn, yn = binned(mid, rhat * sc, lo, hi)
    a.plot(xo, yo, color=GREY, lw=2.0, zorder=4, label="earlier model (shrunk)")
    a.plot(xn, yn, color=BLUE, lw=2.4, zorder=4, label="fastrho (high-$N_e$ fixed)")
    a.set_xlabel("position (kb)"); a.set_ylabel(r"recombination rate ($\times10^{-8}$ c/bp)")
    a.margins(x=0.01); a.set_ylim(0, 3.6)
    a.legend(fontsize=8.2, loc="upper left", framealpha=0.0, handlelength=1.6, borderpad=0.2)
    a.annotate("no anchor:  bias $%.2f\\times$  (was $%.2f\\times$)"
               % (st["overall_br"], st["overall_br_old"]),
               xy=(0.5, 0.045), xycoords="axes fraction", ha="center", fontsize=8.0, va="bottom",
               color="#333", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d0d0d0", alpha=0.9))
    a.set_title("(a) extreme-$N_e$ map ($N_e{=}10^{6}$) recovered in absolute units, no anchor",
                loc="left", fontsize=TITLE_FS)
    ps.panel(a, "a", x=-0.115, y=1.07)

    # ---- (b) the absolute offset is gone: pred/true ratio for the SAME region as (a) ----
    ok = np.isfinite(rhat) & np.isfinite(rtrue) & (rtrue > 0) & (rhat > 0) & (rhat_old > 0)
    lr_old = np.log10(rhat_old[ok] / rtrue[ok]); lr_new = np.log10(rhat[ok] / rtrue[ok])
    bins = np.linspace(-0.9, 0.5, 46)
    b.hist(lr_old, bins=bins, color=GREY, alpha=0.65, label="earlier model", density=True)
    b.hist(lr_new, bins=bins, histtype="step", color=BLUE, lw=2.2, label="fastrho (fixed)", density=True)
    b.axvline(0, color=BLACK, ls=":", lw=1.1)
    b.text(0.0, b.get_ylim()[1] * 0.99, "ideal", ha="center", va="top", fontsize=7.4, color="#777")
    for v, col, ha, dx in ((np.median(lr_old), GREY, "right", -0.03),
                           (np.median(lr_new), BLUE, "left", 0.03)):
        b.axvline(v, color=col, ls="--", lw=1.2)
        b.text(v + dx, b.get_ylim()[1] * 0.62, "%.2f$\\times$" % 10 ** v, ha=ha, fontsize=8.2,
               color=col, fontweight="bold")
    b.set_xlabel("predicted / true rate"); b.set_ylabel("density")
    b.set_xticks(np.log10([0.25, 0.5, 1, 2])); b.set_xticklabels(["0.25", "0.5", "1", "2"])
    b.legend(fontsize=8.0, loc="upper left", framealpha=0.0)
    b.set_title("(b) offset removed (ratio $\\to$ 1)", loc="left", fontsize=TITLE_FS)
    ps.panel(b, "b", x=-0.19, y=1.07)

    # ---- (c) systematic: landscape r stays high, absolute bias moves to ~1 (vs earlier model) ----
    CFG = [("real_drosophila", "Drosophila\n(real Comeron)"), ("anopheles_synth", "Anopheles\n(synthetic)")]
    yy = np.arange(len(CFG))[::-1]
    c.axvline(1.0, color=BLACK, ls=":", lw=1.1)
    c.text(1.0, yy[0] + 0.46, "ideal", ha="center", fontsize=7.4, color="#777")
    for (cfg, lab), y in zip(CFG, yy):
        d = json.load(open(os.path.join(FD, cfg + ".json")))["scales"]
        br_new = np.mean([d[s]["fastrho"]["bias_ratio"] for s in ("25kb", "100kb")])
        br_old = np.mean([d[s].get("fastrho_old", d[s]["fastrho"])["bias_ratio"] for s in ("25kb", "100kb")])
        pe_new = np.mean([d[s]["fastrho"]["pearson"] for s in ("25kb", "100kb")])
        c.plot([br_old, br_new], [y, y], color=STEM, lw=2.6, zorder=1)
        c.scatter([br_old], [y], s=62, color=GREY, edgecolor="white", lw=0.9, zorder=3, marker="D")
        c.scatter([br_new], [y], s=88, color=BLUE, edgecolor="white", lw=0.9, zorder=3, marker="o")
        c.annotate("%.2f$\\times$" % br_new, (br_new, y), xytext=(8, 5), textcoords="offset points",
                   fontsize=8.2, color=BLUE, fontweight="bold")
        c.annotate("%.2f$\\times$" % br_old, (br_old, y), xytext=(-8, -13), textcoords="offset points",
                   fontsize=7.8, color=GREY, ha="center")
        c.annotate("landscape $r$ = %.2f" % pe_new, (0.37, y + 0.24), fontsize=7.6, color="#555")
    c.set_yticks(yy); c.set_yticklabels([lab for _, lab in CFG], fontsize=8.4)
    c.set_xlim(0.35, 1.16); c.set_ylim(-0.55, len(CFG) - 0.20)
    c.set_xlabel("absolute bias ratio (1.0 = ideal)")
    c.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="fastrho (high-$N_e$ fixed)"),
                      Line2D([], [], marker="D", ls="", color=GREY, label="earlier model")],
             fontsize=7.8, loc="lower left", framealpha=0.0)
    c.set_title("(c) systematic: bias $\\to$ 1, landscape preserved", loc="left", fontsize=TITLE_FS)
    ps.panel(c, "c", x=-0.06, y=1.12)

    out = os.path.join(OUT, "fig_dipteran.pdf")
    fig.savefig(out, dpi=600)
    print("wrote", out, "| region bias %.2f (was %.2f)" % (st["overall_br"], st["overall_br_old"]))


if __name__ == "__main__":
    main()
