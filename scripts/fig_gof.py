"""Figure: can we judge an inferred recombination map WITHOUT ground truth?

Four candidate blind signals, each validated against the known true-map Pearson across 4 real
species (human/Drosophila/Arabidopsis/dog). The figure decomposes which failure mode each catches:
 (a) why the naive analog fails  -- the model-free LD-decay observable barely tracks the TRUE map
     (obs_vs_truth), because at mapping resolution LD is dominated by demography/selection, not rate.
 (b) skill of each blind signal at ranking accuracy across species (corr with true Pearson, oriented
     so positive = useful). Reproducibility wins; naive LD is anti-correlated.
 (c) reproducibility (half-vs-half map agreement) vs accuracy -- correctly flags the data-uninformative
     case (dog) as least reproducible; it measures PRECISION, so it misses reproducible BIAS (Drosophila).

Reads <repo>/results/gof_*.json (rsynced from sesame). Saves paper/figures/fig_gof.pdf.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

NAME = {"human": "human", "dmel": "Drosophila", "dog": "dog", "athal": "Arabidopsis"}
COL = ps.SPECIES                   # colorblind-safe: human=blue, dmel=orange, athal=green, dog=brown
ORDER = ["human", "dmel", "athal", "dog"]
ps.style()
plt.rcParams.update({"font.size": 9.5})


def load(name):
    return json.load(open(os.path.join(RES, name)))


def main():
    ld = load("gof_ld.json"); pp = load("gof_pp.json")
    unc = load("gof_uncertainty.json"); rep = load("gof_repro.json")
    truth = {k: ld[k]["truth_pearson"] for k in ld}
    ks = [k for k in ORDER if k in ld]
    tvec = np.array([truth[k] for k in ks])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))

    # (a) the naive observable barely tracks the true map -> a simple blind GoF cannot work
    vals = [ld[k]["obs_vs_truth"] for k in ks]
    ax[0].axhline(0, color="#999", lw=0.9)
    ax[0].vlines(range(len(ks)), 0, vals, color="#dcdcdc", lw=2.4, zorder=1)        # lollipop
    ax[0].scatter(range(len(ks)), vals, s=110, color=[COL[k] for k in ks],
                  edgecolor="white", linewidth=1.0, zorder=3)
    ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels([NAME[k] for k in ks], rotation=18)
    ax[0].set_ylabel("corr(LD-decay observable, true map)")
    ax[0].set_ylim(-0.25, 0.5); ax[0].set_xlim(-0.5, len(ks) - 0.5)
    ax[0].set_title("(a) raw LD is a weak proxy for the map", fontsize=11, loc="left")
    for i, v in enumerate(vals):
        ax[0].text(i, v + (0.03 if v >= 0 else -0.05), "%.2f" % v, ha="center", fontsize=8.5)

    # (b) blind-signal skill at ranking accuracy across species (oriented so + = useful)
    ld_skill = pearsonr([ld[k]["gof"] for k in ks], tvec)[0]            # high gof should = accurate
    pp_skill = pearsonr([pp[k]["gof"] for k in ks], tvec)[0]            # high match should = accurate
    sg_skill = -pearsonr([unc[k]["mean_sigma"] for k in ks], tvec)[0]   # high sigma should = inaccurate
    rp_skill = pearsonr([rep[k]["repro"] for k in ks], tvec)[0]         # high repro should = accurate
    labels = ["naive\nLD-decay", "posterior-\npredictive", "calibrated\nuncertainty", "subsample\nrepro."]
    skills = [ld_skill, pp_skill, sg_skill, rp_skill]
    cols = ["#C44E52" if s < 0 else ps.C["fastrho"] for s in skills]   # lollipop, not bars
    yy = np.arange(4)[::-1]
    ax[1].axvline(0, color="#999", lw=0.9)
    ax[1].hlines(yy, 0, skills, color="#dcdcdc", lw=2.4, zorder=1)
    ax[1].scatter(skills, yy, s=110, color=cols, edgecolor="white", linewidth=1.0, zorder=3)
    ax[1].set_yticks(yy); ax[1].set_yticklabels([l.replace("\n", " ") for l in labels], fontsize=8.5)
    ax[1].set_xlabel("skill: corr with true accuracy")
    ax[1].set_xlim(-1, 1)
    ax[1].set_title("(b) which blind signal predicts accuracy?", fontsize=11, loc="left")
    for yi, v in zip(yy, skills):
        ax[1].annotate("%.2f" % v, (v, yi), xytext=(0, 9), textcoords="offset points",
                       va="bottom", ha="center", fontsize=8.5, color="#333")

    # (c) reproducibility vs accuracy: flags the data-uninformative case (dog), misses bias (fly)
    for k in ks:
        ax[2].scatter(rep[k]["repro"], truth[k], s=130, color=COL[k], edgecolor="k", zorder=3)
        dx = 6 if k != "dog" else -6
        ha = "left" if k != "dog" else "right"
        ax[2].annotate(NAME[k], (rep[k]["repro"], truth[k]), xytext=(dx, 4),
                       textcoords="offset points", fontsize=9, ha=ha)
    ax[2].set_xlabel("subsample reproducibility (half vs half)")
    ax[2].set_ylabel("true-map Pearson (held out)")
    ax[2].set_title("(c) reproducibility = precision  (r=%.2f)" % rp_skill, fontsize=11, loc="left")

    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_gof.pdf")
    fig.savefig(out, bbox_inches="tight"); print("wrote", out)
    print("skills: naiveLD=%.2f pp=%.2f sigma=%.2f repro=%.2f" % tuple(skills))


if __name__ == "__main__":
    main()
