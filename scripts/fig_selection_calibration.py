"""Supplementary: absolute-rate calibration of fastrho vs pyrho under linked selection.

Pearson r (the main-text metric) is scale-invariant and so cannot see multiplicative bias in the
recovered rate. This supplement plots the estimated 25 kb rate against the truth for the neutral
control and the representative hard sweep (matched regions), for both methods. Well-calibrated
estimates track the identity line (slope 1, median est/true 1); a slope < 1 is the usual LD-estimator
regression-to-the-mean, and a vertical offset is multiplicative bias. It shows that fastrho's advantage
under selection is not bought at the cost of absolute-rate bias.

Reads paper/figdata/selection_dr_figdata.npz (calib_* arrays; written by selection_dr_figdata.py).
Run: PYTHONPATH=scripts /home/kkor/venvs/fastrho/bin/python scripts/fig_selection_calibration.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "..", "paper", "figdata", "selection_dr_figdata.npz")
FCOL, PCOL = ps.C["fastrho"], ps.C["pyrho"]
SCALE = 1e8   # plot rates in units of 1e-8 bp^-1


def binned(true, est, nbin=12):
    """Median estimate (+ 25-75% band) of `est` in `nbin` equal-count bins of true rate."""
    lt = np.log10(true)
    edges = np.quantile(lt, np.linspace(0, 1, nbin + 1))
    xc, med, lo, hi = [], [], [], []
    for k in range(nbin):
        m = (lt >= edges[k]) & (lt <= edges[k + 1] if k == nbin - 1 else lt < edges[k + 1])
        if m.sum() < 5:
            continue
        xc.append(10 ** ((edges[k] + edges[k + 1]) / 2))
        q = np.percentile(est[m], [50, 25, 75])
        med.append(q[0]); lo.append(q[1]); hi.append(q[2])
    return (np.array(xc) * SCALE, np.array(med) * SCALE,
            np.array(lo) * SCALE, np.array(hi) * SCALE)


def fit(true, est):
    """log-log OLS slope (ideal 1) and median est/true ratio (ideal 1)."""
    b, _ = np.polyfit(np.log10(true), np.log10(est), 1)
    return b, float(np.median(est / true))


def panel(ax, true, fr, py, title):
    lo = min(true.min(), fr.min(), py.min()) * SCALE
    hi = max(true.max(), fr.max(), py.max()) * SCALE
    ax.plot([lo, hi], [lo, hi], color="#111", ls=(0, (4, 3)), lw=1.1, zorder=1, label="identity")
    for est, col, name in ((fr, FCOL, "fastrho"), (py, PCOL, "pyrho")):
        xc, med, blo, bhi = binned(true, est)
        ax.fill_between(xc, blo, bhi, color=col, alpha=0.15, lw=0)
        b, ratio = fit(true, est)
        ax.plot(xc, med, color=col, marker="o", ms=4.5, mec="white", mew=0.6,
                label=r"%s  (slope %.2f, bias %.2f$\times$)" % (name, b, ratio))
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel(r"true rate ($\times10^{-8}\,$bp$^{-1}$)")
    ax.set_ylabel(r"estimated rate ($\times10^{-8}\,$bp$^{-1}$)")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", fontsize=8.0)
    ax.set_aspect("equal", adjustable="box")


def main():
    ps.style()
    z = np.load(FD, allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.9))
    panel(axes[0], z["calib_true_neutral"], z["calib_fastrho_neutral"], z["calib_pyrho_neutral"],
          "Neutral control")
    panel(axes[1], z["calib_true_sweep"], z["calib_fastrho_sweep"], z["calib_pyrho_sweep"],
          "Hard sweep")
    for a, k in zip(axes, "ab"):
        ps.panel(a, k)
    fig.tight_layout()
    ps.save(fig, "fig_selection_calibration")


if __name__ == "__main__":
    main()
