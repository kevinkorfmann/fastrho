"""Extended Data: fastrho is calibrated, cheap, demography-free and selection-robust.

Consolidates two former ED figures (coverage/cost/demography + linked-selection calibration) into
one four-panel "the estimates are trustworthy" figure, in the shared paper_style.

(a) Interval calibration: empirical vs nominal coverage over 365k held-out intervals (on the
    diagonal; 95.4% at the 95% level).
(b) Cost--accuracy frontier: fastrho is cheapest (one forward pass) AND sharpest at fine scale.
(c) Demography-free: pyrho given a wrong size-history table loses absolute calibration
    (bias ratio 0.78 -> 0.27); fastrho, never told the demography, is unaffected.
(d) Absolute-rate calibration under linked selection (hard sweep): estimated vs true rate; fastrho
    holds its scale better than pyrho (neutral-control slopes/bias in the text).

Reads results_snapshot/{heldout,timings,summary}.json + figdata/selection_dr_figdata.npz.
Run: python scripts/fig_calibration.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RS = os.path.join(_HERE, "paper", "results_snapshot")
FD = os.path.join(_HERE, "paper", "figdata")
OUT = os.path.join(_HERE, "paper", "figures")
ps.style()
C = ps.C
LAB = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}
SCALE = 1e8
HEAD = ["const_n20", "const_n40", "real_hapmap", "real_decode"]


def pear(s, c, sc, m):
    try:
        return float(s[c]["scales"][sc][m]["pearson"])
    except (KeyError, TypeError):
        return np.nan


# ---------------- panels ----------------
def panel_coverage(ax, heldout):
    nom = np.array(heldout["coverage_curve"]["nominal"], float)
    emp = np.array(heldout["coverage_curve"]["empirical"], float)
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="#dfe9f3", alpha=0.7, zorder=0,
                    label="$\\pm$5% band")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.2, label="perfect", zorder=1)
    ax.plot(nom, emp, "-o", color=C["fastrho"], ms=5, markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label="fastrho")
    j = int(np.argmin(np.abs(nom - 0.95)))
    ax.annotate("%.1f%% at the 95%% level" % (emp[j] * 100), (nom[j], emp[j]), xytext=(0.30, 0.82),
                fontsize=8.6, color=C["fastrho"], arrowprops=dict(arrowstyle="->", color=C["fastrho"], lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.legend(loc="upper left", fontsize=8.4)
    ax.set_title("(a) single-pass intervals are calibrated", loc="left", fontsize=9.6)
    ps.panel(ax, "a", x=-0.18)


def panel_cost(ax, s, timings):
    for m in ["fastrho", "pyrho", "relernn"]:
        acc = np.nanmean([pear(s, c, "25kb", m) for c in HEAD])
        au = np.nanmean([s[c]["scales"]["25kb"].get(m, {}).get("hotspot_auprc", np.nan) for c in HEAD])
        cost = timings[m]
        ax.scatter([cost], [acc], s=130, color=C[m], alpha=0.95, edgecolor="white", linewidth=1.3, zorder=3)
        dy = 0.075 if m != "relernn" else -0.14
        ax.annotate(f"{LAB[m]}\n$r$={acc:.2f}, hotspot AUPRC={au:.2f}", (cost, acc),
                    xytext=(cost, acc + dy), ha="center", fontsize=8.2, color=C[m])
    ax.set_xscale("log"); ax.set_xlim(0.4, 3e4); ax.set_ylim(0, 1.0)
    ax.set_xlabel("relative cost per dataset (log)"); ax.set_ylabel("Pearson $r$ (25 kb)")
    ax.text(0.97, 0.05, "cheaper $\\rightarrow$   and sharper $\\uparrow$", transform=ax.transAxes,
            ha="right", fontsize=8.4, color="#888", style="italic")
    ax.set_title("(b) cheapest and sharpest at fine scale", loc="left", fontsize=9.6)
    ps.panel(ax, "b", x=-0.16)


def panel_demog(ax, s):
    def biasr(cfg, m):
        try:
            return s[cfg]["scales"]["100kb"][m].get("bias_ratio", np.nan)
        except KeyError:
            return np.nan
    labs = ["fastrho\n(no demog.)", "pyrho\n(correct table)", "pyrho\n(wrong table)"]
    cols = [C["fastrho"], C["pyrho"], C["pyrho"]]
    vals = [biasr("bottleneck_n20", "fastrho"), biasr("bottleneck_n20", "pyrho"),
            biasr("bottleneck_n20_wd", "pyrho")]
    xx = np.arange(3)
    ax.axhline(1.0, color="#bbb", lw=1.0, ls="--", zorder=1)
    ax.text(2.45, 1.02, "unbiased", fontsize=8, color="#999", ha="right", va="bottom")
    ax.plot([1, 2], [vals[1], vals[2]], color="#cdcdcd", lw=2.2, zorder=1)
    for x_, v, c in zip(xx, vals, cols):
        if v == v:
            ax.scatter(x_, v, s=120, color=c, edgecolor="white", linewidth=1.1, zorder=3)
            ax.annotate("%.2f" % v, (x_, v), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.6, color=c)
    ax.annotate("absolute rate\ncollapses", (2, vals[2]), xytext=(2, vals[2] + 0.30),
                ha="center", fontsize=8.2, color="#777")
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8.6)
    ax.set_xlim(-0.5, 2.6); ax.set_ylim(0, 1.2); ax.set_ylabel("bias ratio (inferred / true rate)")
    ax.grid(axis="x", visible=False)
    ax.set_title("(c) demography-free: pyrho needs the right table, fastrho does not", loc="left",
                 fontsize=9.6)
    ps.panel(ax, "c", x=-0.16)


def _binned(true, est, nbin=12):
    lt = np.log10(true); edges = np.quantile(lt, np.linspace(0, 1, nbin + 1))
    xc, med, lo, hi = [], [], [], []
    for k in range(nbin):
        m = (lt >= edges[k]) & (lt <= edges[k + 1] if k == nbin - 1 else lt < edges[k + 1])
        if m.sum() < 5:
            continue
        xc.append(10 ** ((edges[k] + edges[k + 1]) / 2))
        q = np.percentile(est[m], [50, 25, 75]); med.append(q[0]); lo.append(q[1]); hi.append(q[2])
    return (np.array(xc) * SCALE, np.array(med) * SCALE, np.array(lo) * SCALE, np.array(hi) * SCALE)


def panel_selection(ax, z):
    true, fr, py = z["calib_true_sweep"], z["calib_fastrho_sweep"], z["calib_pyrho_sweep"]
    lo = min(true.min(), fr.min(), py.min()) * SCALE; hi = max(true.max(), fr.max(), py.max()) * SCALE
    ax.plot([lo, hi], [lo, hi], color="#111", ls=(0, (4, 3)), lw=1.1, zorder=1, label="identity")
    for est, col, name in ((fr, C["fastrho"], "fastrho"), (py, C["pyrho"], "pyrho")):
        xc, med, blo, bhi = _binned(true, est)
        b = np.polyfit(np.log10(true), np.log10(est), 1)[0]; ratio = float(np.median(est / true))
        ax.fill_between(xc, blo, bhi, color=col, alpha=0.15, lw=0)
        ax.plot(xc, med, color=col, marker="o", ms=4.5, mec="white", mew=0.6,
                label=r"%s (slope %.2f, bias %.2f$\times$)" % (name, b, ratio))
    lo, hi = 0.1, 15.0   # tighten to the binned-data range (less empty log-log space)
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel(r"true rate ($\times10^{-8}\,$bp$^{-1}$)")
    ax.set_ylabel(r"estimated rate ($\times10^{-8}\,$bp$^{-1}$)")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("(d) absolute-rate calibration under a hard sweep", loc="left", fontsize=9.6)
    ps.panel(ax, "d", x=-0.18)


def main():
    s = json.load(open(os.path.join(RS, "summary.json")))
    heldout = json.load(open(os.path.join(RS, "heldout.json")))
    timings = json.load(open(os.path.join(RS, "timings.json")))
    z = np.load(os.path.join(FD, "selection_dr_figdata.npz"), allow_pickle=True)
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 9.0))
    panel_coverage(ax[0, 0], heldout)
    panel_cost(ax[0, 1], s, timings)
    panel_demog(ax[1, 0], s)
    panel_selection(ax[1, 1], z)
    fig.tight_layout(h_pad=2.2, w_pad=2.4)
    fig.savefig(os.path.join(OUT, "fig_calibration.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig_calibration.png"), dpi=200, bbox_inches="tight")
    print("wrote fig_calibration.pdf")


if __name__ == "__main__":
    main()
