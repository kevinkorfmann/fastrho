"""fig_celegans_specialist.pdf — the C. elegans map inversion + the principled recovery.
(a) On real C. elegans chr I, the frozen model recovers a landscape that is the MIRROR of the meiotic
    (Rockman) map (r=-0.70): its population-LD landscape is inverted by hyperdivergent haplotype blocks
    in the high-recombination arms.
(b) The inversion lives in COMMON variants; SINGLETONS (youngest, structure-immune) recover the correct
    orientation. Recovery r vs allele-count band, at 150 and 500 isotypes -- singletons are the only band
    positive at BOTH sample sizes.
Run (sesame): PYTHONNOUSERSITE=1 python fig_celegans_specialist.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/home/kkor/fastrho_dr/scripts")
import paper_style as ps
ps.style()

D = "/home/kkor/realdata"
OUT = "/home/kkor/fastrho_dr/paper/figures"
BLUE, GREEN, GREY, SLATE = "#1f78b4", "#2f8f4e", "#b9c2c9", "#6b7784"


def z(a):
    a = np.asarray(a, float); return (a - a.mean()) / (a.std() + 1e-12)


tr = json.load(open(f"{D}/transect_celegans.json"))["track"]
c = np.array(tr["centers"], float); pred = z(tr["pred"]); truth = z(tr["truth"])

# measured count-band recoveries (celegans_maf_test.py; count = minor allele count, n-invariant)
bands = ["all", "mac≤1\n(singletons)", "mac≤2", "mac≤3", "mac≤5", "mac≤10", "common\n(≥0.1n)"]
r150 = [-0.703, 0.082, 0.134, 0.137, 0.131, -0.216, -0.699]
r500 = [-0.564, 0.107, -0.242, -0.415, -0.598, -0.647, -0.621]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.5), gridspec_kw=dict(width_ratios=[1.25, 1]))

# (a) the inversion
a1.plot(c, truth, color="#333", lw=1.4, label="Rockman meiotic map")
a1.plot(c, pred, color=BLUE, lw=1.4, label="fastrho (population LD)")
a1.fill_between(c, truth, pred, color=SLATE, alpha=0.15, lw=0)
a1.set_xlabel("Chromosome I position (Mb)"); a1.set_ylabel("recombination rate (z)")
a1.set_title("a  Population-LD map is the mirror of the meiotic map", loc="left", fontsize=10)
a1.text(0.97, 0.05, "r = −0.70", transform=a1.transAxes, ha="right", va="bottom",
        fontsize=11, color=SLATE, fontweight="bold")
a1.legend(loc="upper left", fontsize=7.5, frameon=False)

# (b) count-band recovery
x = np.arange(len(bands)); w = 0.38
for i, (rv, lab, off) in enumerate([(r150, "150 isotypes", -w / 2), (r500, "500 isotypes", w / 2)]):
    cols = [GREEN if v > 0 else SLATE for v in rv]
    a2.bar(x + off, rv, w, color=cols, alpha=0.7 if i == 0 else 1.0,
           edgecolor="white", linewidth=0.5, label=lab)
a2.axhline(0, color="#888", lw=0.8)
a2.axvspan(0.5, 1.5, color=GREEN, alpha=0.08)
a2.set_ylim(-0.78, 0.34)
a2.set_xticks(x); a2.set_xticklabels(bands, fontsize=6.5)
a2.set_ylabel("recovery r  vs meiotic map")
a2.set_title("b  Singletons un-invert the map (both sample sizes)", loc="left", fontsize=10)
a2.annotate("singletons →\ncorrect sign", xy=(1, 0.107), xytext=(2.2, 0.24),
            fontsize=7.5, color=GREEN, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
# legend proxies (colour encodes sign, alpha encodes n)
from matplotlib.patches import Patch
a2.legend(handles=[Patch(fc=GREY, alpha=0.7, label="150 isotypes"),
                   Patch(fc=GREY, alpha=1.0, label="500 isotypes")],
          loc="lower right", fontsize=7, frameon=False)

fig.tight_layout()
ps.save(fig, "fig_celegans_specialist", outdir=OUT, formats=("pdf",))
fig.savefig(f"{OUT}/fig_celegans_specialist.png", dpi=190, bbox_inches="tight", facecolor="white")
print("wrote fig_celegans_specialist")
