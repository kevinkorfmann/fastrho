"""Figure 5 (fig_realdata.pdf): recovery from real genotypes -- identifiability, mechanism, and regime.

Definitive continuous-trajectory design (supersedes the barplot ``fig_realdata.py``).
Every panel is a curve, slope or trajectory -- not a bar.

(a) Identifiability curve: a controlled mutation-rate sweep shows recovery saturating with SNP
    density at a ceiling (~0.8), with a 95% bootstrap band (uncertainty of the pooled estimate
    across replicate maps). Real species are placed by their own diversity -- human sits ON the
    ceiling, but Drosophila/Arabidopsis/wolf/dog carry MORE than enough SNPs yet fall below it:
    their deficit is REGIME, not diversity.
(b) The mechanism (log-y LD decay): each regime distorts the LD decay recombination is read from.
(c) The selfer, as trajectories: pyrho inverts the Arabidopsis map (r=-0.40; a false hotspot in
    the cold pericentromere) while a selfing-aware fastrho recovers the pattern (r=+0.41).
(d) Human (1000G CEU) real recovery: fastrho (r=0.83) and pyrho (r=0.89) both track HapMap.

The dog/wolf "demography is the gate" slopegraph was dropped -- that point is developed fully in
Extended Data Fig. (dog transfer, scripts/fig_dog.py); panel_bottleneck() is kept but unused.

Data resolves locally from paper/figdata (identifiability.json, mechanism.json, realmaps.npz);
regenerate the map bundle with scripts/make_realmaps_bundle.py and the sweep+band with
scripts/identifiability_sweep.py (both on sesame).

Run: PYTHONNOUSERSITE=1 python scripts/fig_identifiability.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL = os.path.join(_HERE, "paper", "figdata")
REAL = _LOCAL if os.path.exists(os.path.join(_LOCAL, "identifiability.json")) else "/home/kkor/realdata"
OUT = os.path.join(_HERE, "paper", "figures")

SP = {"human": ps.SPECIES["human"], "dmel": ps.SPECIES["dmel"], "athal": ps.SPECIES["athal"],
      "wolf": ps.C["fastrho_l"], "dog": ps.SPECIES["dog"]}
NAME = {"human": "human", "dmel": "Drosophila", "athal": "Arabidopsis", "wolf": "wolf", "dog": "dog"}

ps.style()


def _zscore(x):
    """Standardize (mean 0, unit SD). Linear, so it preserves Pearson r exactly."""
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def _smooth(x, w=5):
    if w <= 1:
        return np.asarray(x, float)
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, float), k, mode="same")


def _sat_fit(x, y):
    """Saturating logistic in log10(SNP density): a defensible identifiability ceiling that rises
    then plateaus -- replaces the old cumulative-max hack, and needs no hand-hiding of scatter."""
    lx = np.log10(x)
    try:
        from scipy.optimize import curve_fit

        def logi(t, rmax, k, t0):
            return rmax / (1.0 + np.exp(-k * (t - t0)))

        p, _ = curve_fit(logi, lx, y, p0=[0.83, 3.0, -0.5],
                         bounds=([0.3, 0.3, -2.0], [1.0, 20.0, 2.0]), maxfev=20000)
        xx = np.logspace(lx.min(), lx.max(), 240)
        return xx, logi(np.log10(xx), *p), float(p[0])
    except Exception:  # pragma: no cover
        o = np.argsort(lx)
        return x[o], np.maximum.accumulate(y[o]), float(np.max(y))


# ----------------------------------------------------------------------------- panel (a)
def panel_identifiability(a, d):
    sweep, real = d["sweep"], d["real"]
    xs = np.array([r["snp_per_kb"] for r in sweep], float)
    ys = np.array([r["pearson"] for r in sweep], float)
    lo = np.array([r.get("pearson_lo", np.nan) for r in sweep], float)
    hi = np.array([r.get("pearson_hi", np.nan) for r in sweep], float)
    xx, yy, _ = _sat_fit(xs, ys)
    ceiling = float(max(yy.max(), ys.max()))
    rx_max = max(v["snp_per_kb"] for v in real.values())

    a.axvspan(0.04, 0.25, color="#f3e6e6", alpha=0.8, zorder=0)
    a.text(0.098, 0.045, "power-limited\n(too few SNPs)", fontsize=7.6, color="#a55",
           ha="center", va="bottom")

    # 95% bootstrap band (uncertainty of the pooled recovery across replicate maps)
    if np.isfinite(lo).sum() >= 2:
        o = np.argsort(xs)
        lxx = np.log10(xx); lxs = np.log10(xs[o])
        lo_i = np.interp(lxx, lxs, lo[o]); hi_i = np.interp(lxx, lxs, hi[o])
        a.fill_between(xx, lo_i, hi_i, color="#888", alpha=0.18, lw=0, zorder=1,
                       label="95% CI (bootstrap over maps)")

    a.plot(xx, yy, "-", color="#444", lw=2.2, zorder=3, label="controlled $\\mu$-sweep (fit)")
    a.plot(xs, ys, "o", color="#444", ms=4.5, markeredgecolor="white", zorder=4)
    a.axhline(ceiling, color="#bbb", ls="--", lw=1.0, zorder=1)
    a.text(0.30, ceiling + 0.02, "identifiability ceiling", fontsize=7.8, color="#999",
           va="bottom", ha="left")

    off = {"human": (9, 5, "left"), "dog": (-9, -3, "right"), "wolf": (9, 5, "left"),
           "athal": (11, -1, "left"), "dmel": (11, -1, "left")}
    for k, v in real.items():
        a.scatter(v["snp_per_kb"], v["pearson"], s=115, color=SP[k], edgecolor="k",
                  linewidth=0.8, zorder=5)
        dx, dy, ha = off.get(k, (9, 4, "left"))
        a.annotate(NAME[k], (v["snp_per_kb"], v["pearson"]), xytext=(dx, dy),
                   textcoords="offset points", fontsize=8.3, color=SP[k],
                   fontweight="bold", ha=ha)
        if v["pearson"] < ceiling - 0.08:
            a.annotate("", xy=(v["snp_per_kb"], ceiling - 0.01),
                       xytext=(v["snp_per_kb"], v["pearson"] + 0.02),
                       arrowprops=dict(arrowstyle="->", color=SP[k], lw=1.2, alpha=0.55))

    a.text(0.30, 0.46, "shortfall is set by regime, not diversity:", fontsize=7.8,
           color="#555", style="italic", va="bottom", ha="left")
    causes = [("dog", "domestication bottleneck"), ("athal", "selfing"),
              ("dmel", "In(2L)t inversion"), ("wolf", "coarse pedigree target")]
    for i, (k, txt) in enumerate(causes):
        a.text(0.32, 0.385 - i * 0.078, f"{NAME[k]} — {txt}", fontsize=7.9,
               color=SP[k], va="center", ha="left")

    a.set_xscale("log")
    a.set_xlabel("SNP density (per kb)  $\\propto$  diversity $\\theta = 4N_e\\mu$")
    a.set_ylabel("recovery Pearson $r$")
    a.set_ylim(0, 1.0); a.set_xlim(xs.min() * 0.8, rx_max * 1.5)
    a.legend(loc="lower right", fontsize=8.0)
    a.set_title("recovery saturates with SNP density; real species fall below by regime",
                fontsize=9.6, loc="left")
    ps.panel(a, "a", x=-0.15)


# ----------------------------------------------------------------------------- panel (b)
def panel_mechanism(b, mech):
    reg = [("panmictic", ps.C["fastrho"], "clean decay"),
           ("bottleneck", ps.C["relernn"], "LD inflated at every scale"),
           ("selfing", ps.SPECIES["athal"], "long-range LD retained")]
    for name, col, note in reg:
        dd = mech["ld"][name]
        b.plot(np.array(dd["d"]) / 1000.0, dd["r2"], "-o", color=col, ms=3.2, lw=2.2,
               label=f"{name} — {note}")
    b.set_xscale("log"); b.set_yscale("log")
    b.set_ylim(0.02, 0.85)
    b.set_xlabel("distance between SNPs (kb)"); b.set_ylabel("mean $r^2$ (LD)")
    b.legend(fontsize=7.6, loc="lower left", title="same true recombination rate",
             title_fontsize=7.8, handlelength=1.4)
    b.set_title("each regime distorts the LD decay recombination is read from",
                fontsize=9.6, loc="left")
    ps.panel(b, "b", x=-0.17)


# ----------------------------------------------------------------------------- panel (c)
def panel_bottleneck(c, m):
    x = [0.0, 1.0]
    wolf = [float(m["wolf_rawld"]), float(m["wolf_r"])]
    dog = [float(m["dog_rawld"]), float(m["dog_r"])]
    cw, cd = ps.C["fastrho_l"], ps.SPECIES["dog"]
    c.plot(x, wolf, "-o", color=cw, lw=2.6, ms=10, mec="k", mew=0.6,
           label="wolf (large $N_e$)", zorder=3)
    c.plot(x, dog, "-o", color=cd, lw=2.6, ms=10, mec="k", mew=0.6,
           label="dog (bottlenecked)", zorder=3)
    lab = {(0, "wolf"): (0, 9), (1, "wolf"): (0, 9), (0, "dog"): (0, 9), (1, "dog"): (0, -15)}
    for xi, v in zip(x, wolf):
        c.annotate(f"{v:.2f}", (xi, v), xytext=lab[(int(xi), "wolf")],
                   textcoords="offset points", ha="center", fontsize=8.2, color=cw)
    for xi, v in zip(x, dog):
        c.annotate(f"{v:.2f}", (xi, v), xytext=lab[(int(xi), "dog")],
                   textcoords="offset points", ha="center", fontsize=8.2, color=cd)
    c.annotate("the bottleneck erases the\nmap signal in the data itself", xy=(0, dog[0]),
               xytext=(0.06, 0.145), fontsize=7.4, color=cd, ha="left", va="center",
               arrowprops=dict(arrowstyle="->", color=cd, lw=1.0))
    c.set_xticks(x); c.set_xticklabels(["raw LD $\\leftrightarrow$ map", "fastrho\ninferred"])
    c.set_xlim(-0.4, 1.4); c.set_ylim(0, 0.33)
    c.set_ylabel("Pearson $r$ vs pedigree map")
    c.grid(axis="x", visible=False)
    c.legend(fontsize=8.0, loc="upper right")
    c.set_title("demography is the gate, not the method (dog vs wolf)",
                fontsize=9.6, loc="left")
    ps.panel(c, "c", x=-0.20)


# ----------------------------------------------------------------------------- panel (d)
def _zshow(x, hi=95.0):
    """Standardize for display, but cap the top tail first so a single extreme spike (pyrho puts a
    huge false peak in the pericentromere) cannot flatten the rest of the trajectory to zero."""
    x = np.asarray(x, float)
    x = np.minimum(x, np.percentile(x, hi))
    return _smooth((x - x.mean()) / (x.std() + 1e-12))


def panel_selfer(dax, m):
    cen = m["athal_centers"]
    T, F, P = _zshow(m["athal_truth"]), _zshow(m["athal_pred"]), _zshow(m["athal_pyrho"])
    dax.axhline(0, color="0.8", lw=0.6, zorder=0)
    dax.plot(cen, T, color=ps.C["truth"], lw=2.4, label="Salomé map (truth)", zorder=4)
    dax.plot(cen, F, color=ps.C["fastrho"], lw=1.9, zorder=3, label="fastrho (selfing-aware)")
    dax.plot(cen, P, color=ps.C["pyrho"], lw=1.7, alpha=0.9, zorder=2, label="pyrho (inverts)")
    dax.set_ylim(-2.3, 4.3)   # headroom so pyrho's false-hotspot spike (z~3.8) shows its tip
    ipk = int(np.argmax(P)); itr = int(np.argmin(T))
    dax.annotate("pyrho puts a false hotspot\nin the recombination-cold\npericentromere",
                 xy=(cen[ipk], P[ipk]), xytext=(cen[ipk] + 3.2, 3.7),
                 fontsize=7.2, color=ps.C["pyrho"], ha="left", va="top",
                 arrowprops=dict(arrowstyle="->", color=ps.C["pyrho"], lw=1.1))
    dax.set_xlabel("Arabidopsis chr 1 position (Mb)"); dax.set_ylabel("standardized rate ($z$)")
    dax.legend(fontsize=7.6, loc="lower right")
    dax.set_title("the selfer: pyrho inverts the map; selfing-aware fastrho recovers it",
                  fontsize=9.6, loc="left")
    ps.panel(dax, "c", x=-0.15)


# ----------------------------------------------------------------------------- panel (e)
def panel_human(fig, cell, m):
    cen = np.asarray(m["human_centers"], float)
    tr = np.asarray(m["human_truth"], float)
    fr = np.asarray(m["human_pred"], float)
    py = np.asarray(m["human_pyrho"], float)
    floor = 8e-11
    ytop = max(tr.max(), fr.max(), py.max()) * 1.7
    # Each method is stacked over the same dashed HapMap reference.
    specs = [("fastrho", ps.C["fastrho"], fr, float(m["human_r"])),
             ("pyrho", ps.C["pyrho"], py, float(m["human_pyrho_r"]))]
    sub = cell.subgridspec(2, 1, hspace=0.14)
    for ri, (name, col, yv, rval) in enumerate(specs):
        ax = fig.add_subplot(sub[ri])
        ax.plot(cen, tr, color=ps.C["truth"], lw=0.75, ls=(0, (4, 2.5)), zorder=1)
        ax.plot(cen, yv, color=col, lw=1.6, zorder=3)
        ax.set_yscale("log"); ax.set_ylim(floor, ytop)
        ax.grid(False)
        ax.text(0.012, 0.90, f"{name}   $r={rval:.2f}$", transform=ax.transAxes,
                color=col, fontweight="bold", va="top", ha="left", fontsize=9.2)
        if ri == 0:
            ax.set_xticklabels([])
            ax.set_ylabel("recombination rate (/bp)")
            ax.yaxis.set_label_coords(-0.135, -0.05)   # center across both stacked rows
            ax.set_title("human (1000G CEU): both methods trace the dashed HapMap reference",
                         fontsize=9.6, loc="left")
            ps.panel(ax, "d", x=-0.15, y=1.10)
        else:
            ax.set_xlabel("position (Mb)")


def main():
    d = json.load(open(os.path.join(REAL, "identifiability.json")))
    mech = json.load(open(os.path.join(REAL, "mechanism.json")))
    maps = np.load(os.path.join(_LOCAL, "realmaps.npz"), allow_pickle=True)

    fig = plt.figure(figsize=(13.0, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.24)
    panel_identifiability(fig.add_subplot(gs[0, 0]), d)
    panel_mechanism(fig.add_subplot(gs[0, 1]), mech)
    panel_selfer(fig.add_subplot(gs[1, 0]), maps)
    panel_human(fig, gs[1, 1], maps)

    ps.save(fig, "fig_realdata", outdir=OUT, formats=("pdf",))
    fig.savefig(os.path.join(OUT, "fig_realdata.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
