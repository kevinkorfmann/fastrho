"""I1: the 26-population noise-corrected between-population recombination-divergence matrix.

Spence & Song 2019 (pyrho) inferred fine-scale maps for all 26 1000G populations and reported RAW
between-population correlations, clustering them by continent -- but their own SI S3 concedes that
the low correlations among small-Ne non-African pairs may be "regression attenuation" (estimation
noise), which their point estimates cannot remove. fastrho can: a within-population noise floor
(two disjoint half-samples) turns each raw correlation into an interpretable measurement.

(a) The correction. Left: the RAW between-population correlation matrix (what pyrho reports).
    Right: the NOISE-CORRECTED divergence = floor - between (how far each pair sits BELOW its own
    noise floor). Pairs that look divergent in the raw matrix but sit at their floor are noise.
(b) The S3 test. Across the 325 pairs, the raw between-population correlation tracks the pair's
    noise floor (Spearman rho annotated): low non-African correlations are largely inference noise,
    not biology -- exactly the attenuation pyrho flagged but could not remove.
(c) The floor is set by diversity. Per-population noise floor, coloured by superpopulation:
    lower-diversity (non-African) populations have noisier maps and lower floors, which is WHY
    their raw between-population correlations attenuate.

House style via paper_style. Reads only committed JSON in paper/figdata/.
Build:  python3.13 scripts/fig_between_pop_26.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "scripts")
import paper_style as ps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDATA = os.path.join(HERE, "paper", "figdata")
FIG = os.path.join(HERE, "paper", "figures")
BLUE, LBLUE = ps.C["fastrho"], ps.C["fastrho_l"]
GREY, BLACK = ps.C["relernn"], ps.C["truth"]
ps.style()
plt.rcParams.update({"grid.alpha": 0.18})

# superpopulation display order + colours (ColorBrewer Paired-ish, low-chroma)
SP_ORDER = ["AFR", "EUR", "SAS", "EAS", "AMR"]
SP_COL = {"AFR": ps.CB[1], "EUR": ps.CB[3], "SAS": ps.CB[5], "EAS": ps.CB[7], "AMR": ps.CB[9]}
SCALE = "25kb"


def load(name):
    with open(os.path.join(FIGDATA, name)) as fh:
        return json.load(fh)


def order_pops(pops, superpop):
    idx = sorted(range(len(pops)), key=lambda i: (SP_ORDER.index(superpop[pops[i]]), pops[i]))
    return idx


def main():
    d = load("real_between_pop_26.json")
    arr = np.load(os.path.join(FIGDATA, "real_between_pop_26_arrays.npz"), allow_pickle=True)
    superpop = d["superpop"]
    pops = list(arr["pops"])
    betw = arr[f"betw_{SCALE}"]
    ncdiv = arr[f"ncdiv_{SCALE}"]
    floor = d["scales"][SCALE]["floor_per_pop"]
    o = order_pops(pops, superpop)
    P = [pops[i] for i in o]
    B = betw[np.ix_(o, o)]
    N = ncdiv[np.ix_(o, o)]

    fig = plt.figure(figsize=(11.2, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 1.0], wspace=0.5)

    # ---- (a) raw vs noise-corrected matrices ------------------------------------------
    gsa = gs[0].subgridspec(1, 2, wspace=0.10)
    for k, (M, ttl, cmap, vlab) in enumerate([
            (B, "raw between-pop\ncorrelation (pyrho reports this)", "viridis", "Spearman $r$"),
            (N, "noise-corrected\ndivergence (floor $-$ between)", "magma", "excess drop")]):
        ax = fig.add_subplot(gsa[k])
        M2 = M.copy(); np.fill_diagonal(M2, np.nan)
        im = ax.imshow(M2, cmap=cmap, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl, fontsize=8.2)
        # superpopulation boundaries
        bnd, cur = [], superpop[P[0]]
        for i, p in enumerate(P):
            if superpop[p] != cur:
                bnd.append(i); cur = superpop[p]
        for b in bnd:
            ax.axhline(b - 0.5, color="w", lw=0.8); ax.axvline(b - 0.5, color="w", lw=0.8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5); cb.set_label(vlab, fontsize=7)
        if k == 0:
            # superpop labels down the left
            segs, s0 = [], 0
            for i in range(1, len(P) + 1):
                if i == len(P) or superpop[P[i]] != superpop[P[i - 1]]:
                    segs.append((s0, i - 1, superpop[P[s0]])); s0 = i
            for a, b, sp in segs:
                ax.text(-1.4, (a + b) / 2, sp, rotation=90, va="center", ha="center",
                        fontsize=7, color=SP_COL[sp])
    fig.text(0.055, 0.965, "a", fontsize=12, fontweight="bold")

    # ---- (b) confound control: NCdiv vs Fst, with the identical-map drift baseline -----
    # fastrho reads LD, so allele-frequency divergence (Fst) alone inflates NCdiv even when the
    # true map is IDENTICAL. The dashed baseline is that drift artefact (population splits sharing
    # one map). Real between-continent pairs sit ABOVE it -> genuine map divergence; within-
    # continent pairs cluster at Fst~0, NCdiv~0 (neither drift artefact nor divergence).
    axb = fig.add_subplot(gs[1])
    fp = load("fst_per_pair.json")["pairs"]
    cs = load("confound_split.json")["results"]
    F = np.array([r["fst"] for r in fp]); Y = np.array([r["ncdiv"] for r in fp])
    same = np.array([r["same_continent"] for r in fp])
    axb.scatter(F[same], Y[same], s=10, c=[GREY], alpha=0.55, edgecolor="none",
                label="within-continent", zorder=2)
    axb.scatter(F[~same], Y[~same], s=10, c=[BLUE], alpha=0.55, edgecolor="none",
                label="between-continent", zorder=2)
    bf = np.array([r["fst"] for r in cs]); bn = np.array([r["ncdiv"] for r in cs])
    o = np.argsort(bf)
    axb.plot(bf[o], bn[o], color=BLACK, lw=1.7, ls="--", zorder=3,
             label="drift baseline\n(identical map)")
    axb.axhline(0, color=BLACK, lw=0.6, alpha=0.35, zorder=1)
    axb.set_xlabel("$F_{ST}$  (genetic distance)", fontsize=8)
    axb.set_ylabel("noise-corrected divergence", fontsize=8)
    axb.set_title("real divergence exceeds the drift\nconfound (identical-map control)", fontsize=8.2)
    axb.legend(fontsize=5.9, loc="upper left", frameon=False)
    axb.tick_params(labelsize=6.5)
    fig.text(0.40, 0.965, "b", fontsize=12, fontweight="bold")

    # ---- (c) per-population noise floor, coloured by superpopulation -------------------
    axc = fig.add_subplot(gs[2])
    fp = [(p, floor[p], superpop[p]) for p in P]
    ys = np.arange(len(fp))
    axc.barh(ys, [f for _, f, _ in fp], color=[SP_COL[s] for _, _, s in fp], height=0.8)
    axc.set_yticks(ys); axc.set_yticklabels([p for p, _, _ in fp], fontsize=5.2)
    axc.invert_yaxis()
    axc.set_xlabel("within-population noise floor", fontsize=8)
    axc.set_title("floor set by diversity\n(non-African lower)", fontsize=8.2)
    axc.tick_params(axis="x", labelsize=7)
    axc.set_xlim(min(f for _, f, _ in fp) - 0.03, 1.0)
    fig.text(0.70, 0.965, "c", fontsize=12, fontweight="bold")

    out = os.path.join(FIG, "fig_between_pop_26.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print("wrote", out)
    s = d["scales"][SCALE]
    print(f"[{SCALE}] floor={s['floor_mean']:.3f} between={s['between_mean']:.3f} "
          f"NCdiv within-cont={s['ncdiv_within_continent_mean']:+.3f} "
          f"between-cont={s['ncdiv_between_continent_mean']:+.3f}  "
          f"(panel b: {len(F)} pairs vs drift baseline)")


if __name__ == "__main__":
    main()
