"""Extended Data: trusting an inferred recombination map WITHOUT ground truth.

Unifying thread: subsample reproducibility (inferring on two disjoint halves) is the central blind
diagnostic. (a) Of the candidate blind signals, only reproducibility tracks true accuracy; a naive
LD goodness-of-fit is anti-correlated. (b) Used on a single map, reproducibility flags the
data-limited case (dog) but -- being a measure of precision -- misses reproducible BIAS (the
Drosophila inversion: high reproducibility, only moderate accuracy). (c) The same reproducibility is
the noise floor that separates a true between-population map difference from inference noise.

House-style build: uses paper_style (ps.style fonts, ps.panel bold letters, locked palette) so the
panel matches its Extended Data siblings. Panel (a)'s "skill" values are Pearson correlations over
n=4 species -- stated on the panel so the reader isn't misled by two-decimal precision.

Reads <repo>/results/*.json. Saves paper/figures/fig_trust.pdf.  Build: python3.13 scripts/fig_trust.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "paper", "figures")

# locked palette (single source of truth) -- no ad-hoc hex
BLUE, GREY, BLACK = ps.C["fastrho"], ps.C["relernn"], ps.C["truth"]
RED = ps.CB[4]                       # ColorBrewer "Paired" red (#e31a1c): anti-correlated / misleading
BAND = "#eef1f4"                     # low-chroma neutral wash (matches the blue-grey washes elsewhere)
STEM = "#d9d9d9"                     # light stem grey
SPC = ps.SPECIES                     # human=blue, dmel=orange, athal=green, dog=brown
NAME = {"human": "human", "dmel": "Drosophila", "athal": "Arabidopsis", "dog": "dog"}

ps.style()                           # house fonts (10.5 body / 11.5 title); do NOT override globally
plt.rcParams.update({"grid.alpha": 0.18})

TITLE_FS = 9.6                       # sibling-figure title size (fig_calibration / fig_relernn_scale)


def load(n): return json.load(open(os.path.join(RES, n)))


def main():
    ld, pp = load("gof_ld.json"), load("gof_pp.json")
    unc, rep = load("gof_uncertainty.json"), load("gof_repro.json")
    bp = load("between_pop_d50.json")
    ks = [k for k in ["human", "dmel", "athal", "dog"] if k in ld]
    truth = {k: ld[k]["truth_pearson"] for k in ks}
    tvec = np.array([truth[k] for k in ks])

    fig = plt.figure(figsize=(13.5, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.2], wspace=0.46,
                          left=0.075, right=0.985, bottom=0.145, top=0.86)

    # ------------------------------------------------------------------ (a)
    # which blind signal predicts accuracy? -> subsample reproducibility.
    a = fig.add_subplot(gs[0])
    signal = {
        "subsample reproducibility": pearsonr([rep[k]["repro"] for k in ks], tvec)[0],
        "posterior-predictive":      pearsonr([pp[k]["gof"] for k in ks], tvec)[0],
        "calibrated uncertainty":   -pearsonr([unc[k]["mean_sigma"] for k in ks], tvec)[0],
        "naive LD-decay":            pearsonr([ld[k]["gof"] for k in ks], tvec)[0],
    }
    labs = list(signal)                                  # already ordered best -> most misleading
    vals = [signal[l] for l in labs]
    yy = np.arange(len(labs))[::-1]                      # first entry on top
    win = int(np.argmax(vals))

    a.axvspan(-1, 0, color="#fbeeee", zorder=0)          # anti-correlated half = misleading
    a.axvline(0, color="#bbb", lw=0.9, zorder=1)
    a.hlines(yy, 0, vals, color=STEM, lw=2.8, zorder=2)
    a.scatter(vals, yy,
              s=[168 if i == win else 108 for i in range(len(vals))],
              color=[BLUE if v >= 0 else RED for v in vals],
              edgecolor=[BLACK if i == win else "white" for i in range(len(vals))],
              linewidth=[1.5 if i == win else 1.0 for i in range(len(vals))], zorder=4)
    for y, v in zip(yy, vals):                           # value above each marker (never clips the axis)
        a.annotate("%+.2f" % v, (v, y), xytext=(0, 11), textcoords="offset points",
                   ha="center", va="bottom", fontsize=8.6,
                   color=BLUE if v >= 0 else RED, fontweight="bold")
    a.text(-0.97, -0.62, "anti-correlated = misleading", fontsize=7.4, color="#a76b6b", ha="left")
    a.set_yticks(yy); a.set_yticklabels(labs, fontsize=8.8)
    a.set_xlim(-1, 1); a.set_ylim(-0.75, len(labs) - 0.35)
    a.set_xlabel("skill: correlation with true accuracy")
    a.text(0.99, len(labs) - 0.45, "n = 4 species", fontsize=7.2, color="#888", ha="right", va="top")
    a.set_title("(a) only subsample reproducibility tracks accuracy", fontsize=TITLE_FS, loc="left")
    ps.panel(a, "a", x=-0.42, y=1.10)

    # ------------------------------------------------------------------ (b)
    # reproducibility on one map: flags data-limited, misses reproducible bias.
    b = fig.add_subplot(gs[1])
    b.axvspan(0.93, 0.99, color=BAND, zorder=0)          # the "high reproducibility" column
    # human and Arabidopsis sit at nearly identical reproducibility but 3.5x apart in accuracy:
    xarr = 0.936
    b.annotate("", xy=(xarr, truth["human"]), xytext=(xarr, truth["athal"]),
               arrowprops=dict(arrowstyle="<->", color="#9aa0a6", lw=1.2))
    b.text(0.887, 0.52, "same reproducibility,\naccuracy 0.24–0.83", va="center", ha="center",
           fontsize=7.4, color="#8a8f94")
    for k in ks:
        b.scatter(rep[k]["repro"], truth[k], s=150, color=SPC[k], edgecolor="k",
                  linewidth=0.7, zorder=3)
        b.annotate(NAME[k], (rep[k]["repro"], truth[k]), xytext=(7, 6), textcoords="offset points",
                   fontsize=8.6, color=SPC[k], fontweight="medium")
    b.annotate("data-limited:\nlow reproducibility", (rep["dog"]["repro"], truth["dog"]),
               xytext=(14, 12), textcoords="offset points", fontsize=7.6, color=SPC["dog"])
    b.annotate("reproducible bias\n(inversion): missed", (rep["dmel"]["repro"], truth["dmel"]),
               xytext=(-6, -30), textcoords="offset points", fontsize=7.6, color=SPC["dmel"], ha="center")
    b.set_xlim(0.84, 0.985); b.set_ylim(0.10, 0.92)
    b.set_xlabel("subsample reproducibility (half vs half)"); b.set_ylabel("true accuracy (Pearson $r$)")
    b.set_title("(b) reproducibility is precision, not accuracy", fontsize=TITLE_FS, loc="left")
    ps.panel(b, "b", x=-0.20, y=1.10)

    # ------------------------------------------------------------------ (c)
    # reproducibility = noise floor -> partitions the between-map gap into noise + divergence.
    c = fig.add_subplot(gs[2])
    s = bp["scales"]["25kb"]
    ident, floor, pred = 1.0, s["within_pop_noise_floor"], s["between_pop_pred"]
    c.set_xlim(0.74, 1.035); c.set_ylim(-1.25, 1.85); c.grid(False)
    c.plot([0.755, 1.02], [0, 0], color="#e8e8e8", lw=8, solid_capstyle="round", zorder=0)
    for x, lab, col in [(ident, "identical", BLACK), (floor, "noise floor\n(same pop.)", GREY),
                        (pred, "two pop.\n(fastrho)", BLUE)]:
        c.plot([x], [0], "o", color=col, ms=13, markeredgecolor="white", markeredgewidth=1.6, zorder=4)
        c.plot([x, x], [0, 0.42], color=col, lw=0.9, ls=(0, (2, 2)), zorder=1)   # dropline to bracket
        c.annotate(lab, (x, 0), xytext=(0, -13), textcoords="offset points", ha="center", va="top",
                   fontsize=8.0, color=col)
        c.annotate("%.2f" % x, (x, 0), xytext=(0, 13), textcoords="offset points", ha="center",
                   fontsize=8.8, color=col, fontweight="bold")

    def bracket(x0, x1, yb, txt, col):
        c.plot([x0, x0, x1, x1], [yb - 0.09, yb, yb, yb - 0.09], color=col, lw=1.4)
        c.text((x0 + x1) / 2, yb + 0.08, txt, ha="center", va="bottom", fontsize=8.0, color=col)
    bracket(pred, floor, 1.34, "true divergence\n$%.2f$" % (floor - pred), BLUE)
    bracket(floor, ident, 0.55, "inference noise $%.2f$" % (ident - floor), "#7a7a7a")
    c.set_yticks([]); c.set_xlabel("between-map correlation (25 kb)")
    c.spines["left"].set_visible(False)
    c.set_title("(c) the noise floor separates divergence from noise", fontsize=TITLE_FS, loc="left")
    ps.panel(c, "c", x=-0.09, y=1.10)

    out = os.path.join(FIG, "fig_trust.pdf")
    fig.savefig(out, dpi=600); print("wrote", out, "| skills:", np.round(vals, 2))


if __name__ == "__main__":
    main()
