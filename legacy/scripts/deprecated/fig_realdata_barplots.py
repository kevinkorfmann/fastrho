"""DEPRECATED (2026-07-01) -- superseded by scripts/fig_identifiability.py.

DO NOT render fig_realdata.pdf from this file. It ships the old bar-plot Figure 4/5; the
current, caption-matching figure is produced ONLY by scripts/fig_identifiability.py. (A stale
run of this script during the 15k regen once shipped the wrong fig_realdata.pdf.)

This is the old all-barplot Figure 4. It was retired because three of its four panels were
grouped bar charts (correlation-coefficient "dynamite plots" with no uncertainty), which the
Figure 4 redesign replaced with continuous trajectories: an identifiability ceiling curve, a
log-y LD-decay mechanism, the selfer rescue as truth-vs-prediction trajectories, and the human
real-genotype recovery track. Kept only for provenance. It still writes to fig_realdata.pdf, so
DO NOT run it in the figure build -- it would clobber the current (curve-based) Figure 4.

Original description:
(a) Human recovery track from real 1000G CEU genotypes (truth = HapMap), fastrho vs pyrho.
(b) Real-genotype head-to-head, fastrho vs pyrho: comparable on panmictic phased samples
    (human, Drosophila), but pyrho FAILS on the selfer (Arabidopsis, r<0) and cannot run at all
    on the unphased canid panels.
(c) Demography is the gate, not the method: the dog domestication bottleneck erases the LD->map
    signal (raw-LD<->map ~0), but wolves (large Ne) restore it -> the dog map is recovered.
(d) Mating system is the other gate: a selfing-aware model lifts Arabidopsis from no signal to a
    clear positive correlation, where pyrho is negative.

Run on sesame: PYTHONNOUSERSITE=1 venvs/fastrho/bin/python scripts/fig_realdata.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

MAPS = "/home/kkor/realdata/maps"
OUT = "/home/kkor/fastrho/paper/figures"
C = {"truth": ps.C["truth"], "fastrho": ps.C["fastrho"], "pyrho": ps.C["pyrho"], "fastrho_l": "#9ecae1"}
ps.style()
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})


def load(key):
    return np.load(os.path.join(MAPS, key + ".npz"))


def main():
    gof = json.load(open("/home/kkor/realdata/gof_ld.json"))
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))

    # (a) human recovery track: truth / fastrho / pyrho
    h = load("human"); hp = load("human_pyrho")
    a = ax[0, 0]
    a.plot(h["centers"], h["truth"], color=C["truth"], lw=1.6, label="HapMap (truth)", zorder=3)
    a.plot(h["centers"], h["pred"], color=C["fastrho"], lw=1.1, alpha=0.9,
           label="fastrho  r=%.2f" % float(h["pearson"]))
    a.plot(hp["centers"], hp["pred"], color=C["pyrho"], lw=1.1, alpha=0.9,
           label="pyrho  r=%.2f" % float(hp["pearson"]))
    a.set_yscale("log"); a.set_xlabel("position (Mb)"); a.set_ylabel("recombination rate (/bp)")
    a.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    a.set_title("(a) human (1000G CEU) recovered from real genotypes", loc="left")

    # (b) real-genotype head-to-head bars
    b = ax[0, 1]
    sp = ["human", "Drosophila", "Arabidopsis"]
    fr = [float(load("human")["pearson"]), float(load("dmel")["pearson"]), float(load("athal")["pearson"])]
    py = [float(load("human_pyrho")["pearson"]), float(load("dmel_pyrho")["pearson"]),
          float(load("athal_pyrho")["pearson"])]
    x = np.arange(len(sp)); w = 0.38
    b.bar(x - w / 2, fr, w, color=C["fastrho"], label="fastrho", edgecolor="k", lw=0.5)
    b.bar(x + w / 2, py, w, color=C["pyrho"], label="pyrho", edgecolor="k", lw=0.5)
    b.axhline(0, color="k", lw=0.8)
    for xi, v in zip(x - w / 2, fr): b.text(xi, v + (0.02 if v >= 0 else -0.06), "%.2f" % v, ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, py): b.text(xi, v + (0.02 if v >= 0 else -0.06), "%.2f" % v, ha="center", fontsize=7.5)
    b.set_xticks(x); b.set_xticklabels(sp); b.set_ylabel("Pearson r vs published map")
    b.set_ylim(-0.55, 1.0); b.legend(fontsize=8, loc="lower left")
    b.text(2, 0.62, "pyrho fails\non the selfer", ha="center", fontsize=7.5, color=C["pyrho"])
    b.text(0.5, -0.5, "wolf, dog: pyrho needs phased haplotypes (cannot run)",
           fontsize=7, style="italic", color="0.35")
    b.set_title("(b) head-to-head on real genotypes", loc="left")

    # (c) demography: dog -> wolf
    c = ax[1, 0]
    grp = ["dog\n(bottlenecked breed/village)", "wolf\n(large $N_e$)"]
    raw = [gof["dog"]["obs_vs_truth"], gof["wolf"]["obs_vs_truth"]]
    inf = [float(load("dog")["pearson"]), float(load("wolf")["pearson"])]
    x = np.arange(2)
    c.bar(x - w / 2, raw, w, color="0.6", label="raw LD $\\leftrightarrow$ map", edgecolor="k", lw=0.5)
    c.bar(x + w / 2, inf, w, color=C["fastrho"], label="fastrho inferred", edgecolor="k", lw=0.5)
    for xi, v in zip(x - w / 2, raw): c.text(xi, v + 0.008, "%.2f" % v, ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, inf): c.text(xi, v + 0.008, "%.2f" % v, ha="center", fontsize=7.5)
    c.set_xticks(x); c.set_xticklabels(grp, fontsize=8); c.set_ylabel("Pearson r vs Campbell map")
    c.set_ylim(0, 0.35); c.legend(fontsize=8, loc="upper left")
    c.set_title("(c) the bottleneck erases the signal; wolves restore it", loc="left")

    # (d) selfing: Arabidopsis rescue
    d = ax[1, 1]
    labels = ["pyrho", "fastrho\n(base model)", "fastrho\n(selfing-aware)"]
    vals = [float(load("athal_pyrho")["pearson"]), -0.078, float(load("athal")["pearson"])]
    cols = [C["pyrho"], C["fastrho_l"], C["fastrho"]]
    xb = np.arange(3)
    d.bar(xb, vals, 0.6, color=cols, edgecolor="k", lw=0.5)
    d.axhline(0, color="k", lw=0.8)
    for xi, v in zip(xb, vals): d.text(xi, v + (0.02 if v >= 0 else -0.06), "%.2f" % v, ha="center", fontsize=8)
    d.set_xticks(xb); d.set_xticklabels(labels, fontsize=8); d.set_ylabel("Pearson r vs Salomé map")
    d.set_ylim(-0.55, 0.6)
    d.set_title("(d) a selfing-aware model recovers Arabidopsis", loc="left")

    fig.tight_layout()
    out = os.path.join(OUT, "fig_realdata.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
