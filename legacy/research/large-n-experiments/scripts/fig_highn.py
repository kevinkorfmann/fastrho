"""Large-$n$ scaling, and fastrho's STANDALONE amortized composite likelihood matching pyrho
on real human data with NO pyrho binary and NO per-dataset two-locus table at inference.

(a) On the neutral sim benchmark the amortized SSM leads pyrho at small $n$, but its
    sample-size-robust features saturate while pyrho's exact two-locus likelihood keeps
    improving, so pyrho overtakes it around $n\\sim100$ -- and real single-population human
    panels are larger ($n=198$ here). This is a structural amortization-vs-exact gap, and it
    is why the amortized SSM alone cannot match pyrho at high $n$: it motivates fastrho's own
    composite-likelihood readout.
(b) On real 1000G high-coverage GRCh38 CEU data vs the deCODE pedigree map (selection-immune,
    12 loci, 100 kb), fastrho's amortized composite likelihood (CL-map: a fused-LASSO readout
    on ONE fixed, demography-robust two-locus table, data-derived Ne, no pyrho, no per-dataset
    table) sits right on top of pyrho at every neutral locus (neutral mean 0.84 vs 0.84), while
    fastrho's SSM (rich) dominates the SLC24A5 selective sweep where both collapse (0.71 vs 0.55).
    A standalone regime-aware pi-gate (CL-map at neutral windows, SSM at low-diversity sweep
    windows) beats pyrho at BOTH regimes.

Reads paper/figdata/highn.npz.  Run: PYTHONPATH=scripts python3.13 scripts/fig_highn.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "..", "paper", "figdata", "highn.npz")

SSM = ps.C["fastrho_l"]   # light blue -- fastrho amortized SSM (rich)
CL = ps.C["fastrho"]      # dark blue  -- fastrho composite likelihood (CL-map): the standalone match
PYR = ps.C["pyrho"]       # green      -- pyrho
SSM_E = "#2b7bba"         # darker edge so the light-blue SSM markers stay legible

ps.style()
plt.rcParams.update({"axes.titlesize": 10.5, "font.size": 9.5})


def main():
    d = np.load(FD, allow_pickle=True)
    loci, kind = d["loci"], d["loci_kind"]
    rich, clmap, pyr = d["loci_rich"], d["loci_clmap"], d["loci_pyrho"]
    gneu, gsw = float(d["gate_neutral"]), float(d["gate_sweep"])
    neu = kind != "sweep"
    cl_neu, py_neu = clmap[neu].mean(), pyr[neu].mean()

    fig = plt.figure(figsize=(12.6, 5.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.62], top=0.86, bottom=0.135,
                          left=0.065, right=0.99, wspace=0.24)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])

    # ---- (a) sample-size crossover on sims (why the amortized SSM alone cannot win at large n) ----
    n = d["ncross_n"]
    axa.axvspan(90, 230, color=ps.HIGHLIGHT, zorder=0)
    axa.plot(n, d["ncross_base"], color=SSM, marker="o", ms=7.0, mec=SSM_E, mew=1.1,
             lw=2.6, label="fastrho (amortized SSM)")
    axa.plot(n, d["ncross_pyrho"], color=PYR, marker="s", ms=6.5, mec="white", mew=0.8,
             lw=2.4, label="pyrho")
    axa.axvline(198, color="#888", ls=(0, (3, 3)), lw=1.0)
    axa.annotate("real human\n$n{=}198$", (198, 0.752), ha="center", va="bottom",
                 fontsize=7.8, color="#555")
    axa.annotate("pyrho\novertakes", (150, 0.905), ha="center", va="center",
                 fontsize=8.0, color="#8a6d3b")
    axa.set_xscale("log")
    axa.set_xticks([20, 40, 100, 200])
    axa.set_xticklabels([20, 40, 100, 200])
    axa.set_xlim(17, 230)
    axa.set(xlabel="sample size $n$ (haplotypes)", ylabel="Pearson $r$ (25 kb)", ylim=(0.72, 0.98))
    axa.legend(loc="lower right", fontsize=8.6)
    axa.set_title("Amortized SSM features saturate at large $n$", loc="left")

    # ---- (b) real 12-locus recovery @100kb: SSM vs standalone CL-map vs pyrho ---------------------
    order = np.argsort(pyr)                        # hardest (lowest pyrho) at the bottom
    y = np.arange(len(loci))
    swy = int(np.where(order == np.where(~neu)[0][0])[0][0])   # row of the sweep locus (index 0)

    # neutral "match" ties: thin grey link between CL-map and pyrho (they overlap -> reads as matched)
    for yi, i in zip(y, order):
        if neu[i]:
            axb.plot([clmap[i], pyr[i]], [yi, yi], color="#c9c9c9", lw=1.4, zorder=1,
                     solid_capstyle="round")
    # sweep row: arrow from pyrho up to the SSM (SSM >> pyrho)
    axb.annotate("", xy=(rich[~neu][0], swy), xytext=(pyr[~neu][0], swy),
                 arrowprops=dict(arrowstyle="-|>", color=SSM_E, lw=1.8,
                                 shrinkA=3, shrinkB=3), zorder=2)

    axb.scatter(rich[order], y, s=54, color=SSM, ec=SSM_E, lw=0.9, zorder=3,
                label="fastrho SSM (amortized)")
    axb.scatter(pyr[order], y, s=82, color=PYR, marker="D", ec="white", lw=0.7, zorder=4,
                label="pyrho")
    axb.scatter(clmap[order], y, s=34, color=CL, ec="white", lw=0.7, zorder=5,
                label="fastrho composite likelihood")

    # neutral-mean vlines: CL-map and pyrho coincide
    axb.axvline(py_neu, color=PYR, ls=(0, (2, 2)), lw=1.5, zorder=0)
    axb.axvline(cl_neu, color=CL, ls=(1.5, (2, 2)), lw=1.5, zorder=0)
    axb.annotate("neutral means\ncoincide (%.2f)" % cl_neu, (cl_neu, len(loci) - 0.35),
                 xytext=(cl_neu - 0.075, len(loci) - 0.05), fontsize=7.6, color="#333",
                 ha="center", va="top",
                 arrowprops=dict(arrowstyle="-", color="#999", lw=0.8))

    # sweep callout (placed in the empty bottom-left, clear of the gate note)
    axb.annotate("SLC24A5 sweep:\nfastrho SSM $\\gg$ pyrho", (0.215, swy),
                 fontsize=8.0, color=SSM_E, va="center", ha="left", fontweight="normal")

    # secondary regime-aware gate note (a frontier, NOT a free lunch), compact, bottom-right
    axb.text(0.985, 0.035,
             "regime-aware $\\pi$-gate (SSM in low-diversity windows):\n"
             "recovers the SSM's sweep gain over pyrho (%.3f vs %.3f)\n"
             "at a small neutral cost (%.3f, vs pyrho %.3f)" % (gsw, pyr[~neu][0], gneu, py_neu),
             transform=axb.transAxes, ha="right", va="bottom", fontsize=7.2, color="#222",
             bbox=dict(boxstyle="round,pad=0.42", fc="#f4f6f8", ec="#c2ccd6", lw=0.7))

    axb.set_yticks(y)
    axb.set_yticklabels([f"{loci[i]}" + (" $*$" if not neu[i] else "") for i in order], fontsize=8.4)
    axb.set_ylim(-0.75, len(loci) - 0.25)
    axb.set(xlabel="Pearson $r$ vs deCODE pedigree map (100 kb)", xlim=(0.20, 1.0))
    axb.legend(loc="upper left", fontsize=8.2, handletextpad=0.4, borderaxespad=0.6)
    axb.set_title("fastrho's amortized composite likelihood matches pyrho, no per-dataset table  "
                  "(neutral %.2f vs %.2f);  its SSM dominates the sweep (%.2f vs %.2f)"
                  % (cl_neu, py_neu, rich[~neu][0], pyr[~neu][0]),
                  loc="left", fontsize=9.2)

    ps.panel(axa, "a")
    ps.panel(axb, "b", x=-0.135)
    ps.save(fig, "fig_highn")


if __name__ == "__main__":
    main()
