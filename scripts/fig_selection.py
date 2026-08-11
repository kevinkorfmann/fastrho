"""Linked-selection robustness figure (main-text) -- densely-sampled dose-response.

Six panels from the SLiM stress test (frozen base model, neutral training prior, no retraining;
Ne=1e4, n=20, 40 regions per condition; region-bootstrap 95% CIs). Conditions are discovered from
their selection parameters, so the curves densify automatically as more are added.
  (a) diversity footprint of selection             (d) recovery vs sweep strength (2Ne*s)
  (b) absolute-rate accuracy: pyrho collapses,      (e) recovery vs background-selection intensity
      fastrho holds (neutral -> sweep bias)         (f) paired fastrho-minus-pyrho gap, all conditions
  (c) recovery vs distance from the swept site

Narrative order: selection *distorts* the data (a); yet fastrho still recovers the absolute rate
while pyrho's estimate collapses to ~half-true under the sweep (b, the payoff Pearson r cannot see);
spatially the loss is confined to the diversity valley (c); it scales gently with dose (d, e); and the
paired advantage over pyrho holds across all 24 conditions (f).

Panel b distils the Extended-Data calibration (fig:selcalib) to its single most striking result: the
median estimated/true rate ratio (log scale, 1 = unbiased) for each method in the neutral control vs
the hard sweep, with per-window IQR. pyrho is the better-calibrated estimator under neutrality but
crosses *below* fastrho under selection, collapsing to ~2x low. Panels d, e shade the fastrho-pyrho
advantage as a filled band so the gain is seen, not inferred. A single shared method legend (top)
serves a, c, d, e; the grey dashed "neutral reference" is the pooled 25 kb neutral Pearson r in d/e and
the neutral log-rate accuracy curve in c.

Reads paper/figdata/selection_dr.json + selection_dr_figdata.npz.
Run: PYTHONPATH=scripts python scripts/fig_selection.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paper_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
DR = os.path.join(HERE, "..", "paper", "figdata", "selection_dr.json")
FD = os.path.join(HERE, "..", "paper", "figdata", "selection_dr_figdata.npz")
NE = 1e4
FCOL, PCOL = ps.C["fastrho"], ps.C["pyrho"]
GREY = ps.REGIME["neutral"]

ps.style()
plt.rcParams.update({"axes.titlesize": 10, "font.size": 9.5})


def is_sweep(r):
    return r["demography"] == "slim_sweep" and int(r.get("soft_k") or 1) == 1


def line_with_ci(ax, xs, recs, key_pref, col, label, shade_vs=None):
    """Plot pooled Pearson (with bootstrap CI band) for one method over an x-series.

    Returns (X, M) so a caller can shade the advantage against another method's curve.
    """
    pts = []
    for x, r in zip(xs, recs):
        v = next((r[k] for k in (key_pref + "_cmn_25kb", key_pref + "_25kb") if r.get(k)), None)
        if v:
            pts.append((x, v))
    if not pts:
        return None
    X = np.array([p[0] for p in pts])
    M = np.array([p[1][0] for p in pts])
    lo = np.array([p[1][1] for p in pts])
    hi = np.array([p[1][2] for p in pts])
    ax.fill_between(X, lo, hi, color=col, alpha=0.15, lw=0)
    ax.plot(X, M, color=col, marker="o", ms=4.5, mec="white", mew=0.6, label=label, zorder=4)
    return X, M


def bias_stats(z, tag, method):
    """Per-window estimated/true rate ratio -> (median, q25, q75)."""
    r = z["calib_%s_%s" % (method, tag)] / z["calib_true_%s" % tag]
    return float(np.median(r)), float(np.percentile(r, 25)), float(np.percentile(r, 75))


def main():
    d = json.load(open(DR))
    recs = d["conditions"]
    by = {r["name"]: r for r in recs}
    neutral = by.get("neutral")
    n0 = neutral["mean_sites"]
    ref = neutral["fastrho_25kb"][0]
    z = np.load(FD, allow_pickle=True)

    # dose-response series (hoisted so d, e can share one y-range for fair visual comparison)
    sw = sorted([r for r in recs if is_sweep(r) and (r.get("sweep_target") or 1) >= 0.999],
                key=lambda r: r["sweep_s"])
    bg = sorted([r for r in recs if r["demography"] == "slim_bgs"],
                key=lambda r: r["mean_sites"], reverse=True)

    def yext(*groups):
        los, his = [], []
        for g in groups:
            for r in g:
                for kp in ("fastrho", "pyrho"):
                    v = next((r[k] for k in (kp + "_cmn_25kb", kp + "_25kb") if r.get(k)), None)
                    if v:
                        los.append(v[1]); his.append(v[2])
        return (min(los) - 0.02, max(his) + 0.02)
    SHY = yext(sw, bg)   # common (lo, hi) for panels d, e

    fig = plt.figure(figsize=(12.6, 7.4))
    gs = fig.add_gridspec(2, 3, top=0.88, bottom=0.075, left=0.065, right=0.985,
                          hspace=0.42, wspace=0.31)
    ax = {k: fig.add_subplot(gs[i, j]) for k, (i, j) in
          {"a": (0, 0), "b": (0, 1), "c": (0, 2), "d": (1, 0), "e": (1, 1), "f": (1, 2)}.items()}

    def refline(a):
        a.axhline(ref, color=GREY, ls=(0, (4, 3)), lw=1.0, zorder=0)

    def shade_gain(a, Xf, Mf, Mp):
        """Fill the fastrho-over-pyrho advantage so the gain reads as an area."""
        a.fill_between(Xf, Mp, Mf, where=Mf >= Mp, color=FCOL, alpha=0.13, lw=0, zorder=1)

    # (a) diversity footprint (mean over 40 regions; shaded = 95% CI of the mean)
    x = z["centres"] / 1e6
    for tag, lab in (("neutral", "neutral"), ("bgs", "background selection"), ("sweep", "hard sweep")):
        if "pi_%s" % tag in z.files:
            col = ps.REGIME[tag]
            if "pi_%s_lo" % tag in z.files:
                ax["a"].fill_between(x, z["pi_%s_lo" % tag] * 1e4, z["pi_%s_hi" % tag] * 1e4,
                                     color=col, alpha=0.18, lw=0)
            ax["a"].plot(x, z["pi_%s" % tag] * 1e4, color=col, lw=1.6, label=lab)
    ax["a"].axvspan(0.75, 1.25, color=ps.HIGHLIGHT, zorder=0)
    ax["a"].set(xlabel="position (Mb)", ylabel=r"diversity $\pi$ ($\times10^{-4}$)", ylim=(0, 7.6))
    ax["a"].legend(loc="lower left", fontsize=7.6, title="selection regime", title_fontsize=7.8)
    ax["a"].set_title("Selection distorts the data", loc="left")

    # (b) MONEY SHOT: absolute-rate accuracy Pearson r cannot see. Median estimated/true rate ratio
    #     (1 = unbiased) for each method in the neutral control vs the hard sweep, with per-window IQR.
    #     pyrho is better calibrated when neutral but crosses BELOW fastrho under the sweep (~2x low).
    axb = ax["b"]
    regimes = [("neutral", "neutral"), ("sweep", "hard\nsweep")]
    xpos = {"neutral": 0.0, "sweep": 1.0}
    dxm = {"fastrho": -0.13, "pyrho": 0.13}   # dodge the two methods within each regime
    axb.axhline(1.0, color="#111", ls=(0, (4, 3)), lw=1.0, zorder=1)
    axb.text(-0.45, 1.005, "unbiased (=1)", ha="left", va="bottom", fontsize=7.4, color="#555")
    for method, col in (("fastrho", FCOL), ("pyrho", PCOL)):
        xs, meds = [], []
        for tag, _ in regimes:
            med, q25, q75 = bias_stats(z, tag, method)
            xx = xpos[tag] + dxm[method]
            axb.add_line(Line2D([xx, xx], [q25, q75], color=col, lw=5.0, alpha=0.20,
                                solid_capstyle="round", zorder=2))  # IQR band (window spread)
            axb.plot(xx, med, "o", ms=7, color=col, mec="white", mew=1.0, zorder=4)
            xs.append(xx); meds.append(med)
        axb.plot(xs, meds, color=col, lw=1.6, zorder=3)   # neutral->sweep trajectory (dumbbell)
    # call out the collapse vs the hold
    pm = bias_stats(z, "sweep", "pyrho")[0]
    fm = bias_stats(z, "sweep", "fastrho")[0]
    axb.annotate(r"$\times%.2f$" % pm + "\n(~2$\\times$ low)", (1 + dxm["pyrho"], pm),
                 xytext=(10, -2), textcoords="offset points", ha="left", va="center",
                 fontsize=7.8, color=PCOL, fontweight="bold")
    axb.annotate(r"$\times%.2f$" % fm, (1 + dxm["fastrho"], fm), xytext=(-8, 8),
                 textcoords="offset points", ha="right", va="bottom", fontsize=7.8,
                 color=FCOL, fontweight="bold")
    axb.set_xlim(-0.5, 1.55)
    axb.set_xticks([0, 1]); axb.set_xticklabels(["neutral", "hard sweep"])
    axb.set_ylim(0.42, 1.06)
    axb.set_ylabel(r"estimated / true rate")
    axb.grid(axis="x", visible=False)
    axb.set_title("Absolute-rate accuracy under selection", loc="left")

    # (c) recovery vs distance from swept site: fastrho vs pyrho on the hard sweep, with CI bands
    if "dist_bins" in z.files:
        db = z["dist_bins"]
        ax["c"].axvspan(0, 0.16, color=ps.HIGHLIGHT, zorder=0)
        ax["c"].fill_between(db, z["recov_sweep_lo"], z["recov_sweep_hi"], color=FCOL, alpha=0.15, lw=0)
        ax["c"].plot(db, z["recov_sweep"], color=FCOL, marker="o", ms=3.5, lw=1.8, label="fastrho")
        if "recov_pyrho_sweep" in z.files:
            ax["c"].fill_between(db, z["recov_pyrho_sweep_lo"], z["recov_pyrho_sweep_hi"],
                                 color=PCOL, alpha=0.15, lw=0)
            ax["c"].plot(db, z["recov_pyrho_sweep"], color=PCOL, marker="o", ms=3.5, lw=1.8,
                         label="pyrho")
        ax["c"].plot(db, z["recov_neutral"], color=GREY, ls=(0, (4, 3)), lw=1.2,
                     label="neutral reference")
        ymin = float(min(z["recov_sweep_lo"].min(), z["recov_pyrho_sweep_lo"].min())) - 0.03
    else:
        ymin = 0.30
    ax["c"].set(xlabel="distance from swept site (Mb)", ylabel="Pearson $r$ (log rate)",
                ylim=(ymin, 0.95))
    ax["c"].set_title("Recovery vs. distance from sweep", loc="left")

    # (d) sweep strength: all hard sweeps (target=1); pyrho overlaid, advantage shaded
    xb = [2 * NE * r["sweep_s"] for r in sw]
    fb = line_with_ci(ax["d"], xb, sw, "fastrho", FCOL, "fastrho")
    pb = line_with_ci(ax["d"], xb, sw, "pyrho", PCOL, "pyrho")
    if fb and pb:
        shade_gain(ax["d"], fb[0], fb[1], pb[1])
    refline(ax["d"]); ax["d"].set_xscale("log")
    ax["d"].set(xlabel=r"sweep strength  $2N_e s$", ylabel="Pearson $r$ (25 kb)", ylim=SHY)
    ax["d"].set_title("Recovery vs. sweep strength", loc="left")

    # (e) background-selection intensity vs realized diversity reduction, advantage shaded
    xc = [100 * (1 - r["mean_sites"] / n0) for r in bg]
    fc = line_with_ci(ax["e"], xc, bg, "fastrho", FCOL, "fastrho")
    pc = line_with_ci(ax["e"], xc, bg, "pyrho", PCOL, "pyrho")
    if fc and pc:
        shade_gain(ax["e"], fc[0], fc[1], pc[1])
    refline(ax["e"])
    ax["e"].set(xlabel="diversity reduction (%)", ylabel="Pearson $r$ (25 kb)", ylim=SHY)
    ax["e"].set_title("Recovery vs. background selection", loc="left")

    # (f) paired advantage: Delta r = r(fastrho) - r(pyrho) at 25 kb, per condition, with paired CI
    blocks = [
        ("sweep\nstrength", sorted([r for r in recs if r["name"].startswith("swstr_")
                                    and r.get("delta_25kb")], key=lambda r: r["sweep_s"])),
        ("background\nselection", sorted([r for r in recs if r["name"].startswith("bgsint_")
                                          and r.get("delta_25kb")], key=lambda r: -r["mean_sites"])),
        ("sweep\ncompleteness", sorted([r for r in recs if r["name"].startswith("compl_")
                                        and r.get("delta_25kb")], key=lambda r: r["sweep_target"])),
    ]
    axf = ax["f"]
    axf.axvline(0, color="#888", ls=(0, (4, 3)), lw=1.0, zorder=1)
    y = 0.0
    ticks, ticklabels, allhi = [], [], []
    for bi, (bname, brecs) in enumerate(blocks):
        ys = []
        for r in brecs:
            y -= 1.0
            ys.append(y)
            dlt, lo, hi = r["delta_25kb"]
            allhi.append(hi)
            sig = lo > 0
            axf.errorbar(dlt, y, xerr=[[dlt - lo], [hi - dlt]], fmt="o", ms=4.6, color=FCOL,
                         mfc=FCOL if sig else "white", mec=FCOL, mew=1.0,
                         elinewidth=1.1, capsize=2.0, zorder=3)
        if ys:
            axf.axhspan(min(ys) - 0.5, max(ys) + 0.5, color="#f4f4f4", zorder=0)
            ticks.append(np.mean(ys)); ticklabels.append(bname)
        y -= 0.8
    axf.set_yticks(ticks); axf.set_yticklabels(ticklabels, fontsize=8.2)
    axf.set_ylim(y + 0.4, 0.0)
    xhi = max(allhi) + 0.03
    axf.set_xlim(-0.03, xhi)
    axf.set_xlabel(r"$\Delta r$  (fastrho $-$ pyrho), 25 kb")
    axf.grid(axis="y", visible=False)
    axf.annotate("fastrho better " + r"$\rightarrow$", (xhi, 0.15), xycoords=("data", "axes fraction"),
                 ha="right", va="bottom", fontsize=7.8, color="#555")
    axf.set_title("Paired advantage over pyrho", loc="left")

    # --- one shared method legend for panels a, c, d, e (removes repeated per-panel legends) ---
    mh = [Line2D([0], [0], color=FCOL, marker="o", ms=5, mec="white", mew=0.6, lw=2.2, label="fastrho"),
          Line2D([0], [0], color=PCOL, marker="o", ms=5, mec="white", mew=0.6, lw=2.2, label="pyrho"),
          Line2D([0], [0], color=GREY, ls=(0, (4, 3)), lw=1.4, label="neutral reference")]
    fig.legend(handles=mh, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3,
               frameon=False, fontsize=10, handlelength=2.2, columnspacing=2.4)

    for k in ax:
        ps.panel(ax[k], k)
    ps.save(fig, "fig_selection")


if __name__ == "__main__":
    main()
