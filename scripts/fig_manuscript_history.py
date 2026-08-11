"""Manuscript Figure 2: evolutionary history changes what LD can reveal.

This cache-only renderer restores the biological trajectories that were lost in
the first compact composite.  The top band begins with empirical dog and wolf
agreement with the same pedigree reference, then follows a canid bottleneck from
LD inflation to loss of a shared map and population transfer.  The lower band shows the complete five-chromosome
Arabidopsis maps and validates the selfing-aware rescue against two independent
meiotic references.

No simulation or inference is performed here.  Every plotted quantity is read
from committed figure-data archives.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np

import paper_style as ps


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDATA = os.path.join(ROOT, "paper", "figdata")
OUT = os.path.join(ROOT, "paper", "manuscript", "figures")
CHROMS = ("1", "2", "3", "4", "5")

BLUE = "#2737E7"
LIGHT_BLUE = "#6F7CF0"
GREEN = "#151515"
ORANGE = "#777777"
GRAY = "#777777"
BLACK = "#151515"
BOUNDARY = "#8A8A8A"


def _panel(ax, letter, x=-0.17, y=1.08):
    ps.panel(ax, letter, x=x, y=y, fontsize=13)


def _pearson_log(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return float(np.corrcoef(np.log(a[keep]), np.log(b[keep]))[0, 1])


def _zmask(x, mask):
    x = np.asarray(x, float)
    z = np.full_like(x, np.nan)
    z[mask] = (x[mask] - x[mask].mean()) / x[mask].std()
    return z


def _smooth(x, width=5):
    x = np.asarray(x, float)
    valid = np.isfinite(x).astype(float)
    filled = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(width)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(denominator > 0, numerator / denominator, np.nan)
    out[~np.isfinite(x)] = np.nan
    return out


def _pericentromere(centers, ztruth, mask, width=11, percentile=30):
    """Return the contiguous central trough of the smoothed meiotic map."""
    smooth = _smooth(ztruth, width)
    n = len(centers)
    idx = np.arange(n)
    good = np.isfinite(smooth) & mask
    central = good & (idx >= int(0.15 * n)) & (idx <= int(0.85 * n))
    if not central.any():
        return None, None
    minimum = idx[central][np.argmin(smooth[central])]
    threshold = np.nanpercentile(smooth[good], percentile)
    lo = hi = int(minimum)
    while lo > 0 and np.isfinite(smooth[lo - 1]) and smooth[lo - 1] < threshold:
        lo -= 1
    while hi + 1 < n and np.isfinite(smooth[hi + 1]) and smooth[hi + 1] < threshold:
        hi += 1
    return lo, hi


def panel_empirical_canids(ax, empirical):
    scale = np.asarray(empirical["window_kb"], float)
    wolf = np.asarray(empirical["wolf_pearson"], float)
    dog = np.asarray(empirical["dog_pearson"], float)
    village = np.asarray(empirical["village_dog_pearson"], float)
    ax.plot(scale, wolf, "o-", color=BLUE, lw=2.1, ms=5.0,
            label="wolf transfer ($n=33$)")
    ax.plot(scale, dog, "s-", color=ORANGE, lw=2.0, ms=4.7,
            label="all dogs ($n=67$)")
    ax.plot(scale, village, "^-", color="#B6B6B6", lw=1.7, ms=4.5,
            mfc="white", mew=1.0, label="village dogs ($n=42$)")
    ax.set_xscale("log")
    ax.set_xlim(82, 2400)
    ax.set_ylim(0.10, 0.66)
    ax.xaxis.set_major_locator(mticker.FixedLocator([100, 200, 500, 1000, 2000]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["100", "200", "500", "1,000", "2,000"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("comparison window (kb)")
    ax.set_ylabel("Pearson $r$ with pedigree map")
    ax.set_title("Wolf LD modestly improves\na difficult interval", loc="left", fontsize=9.0)
    ax.legend(fontsize=6.7, loc="upper left", handlelength=1.35)
    ax.annotate(f"{wolf[3]:.2f}", (scale[3], wolf[3]), xytext=(2, 7),
                textcoords="offset points", color=BLUE, fontsize=7.6, fontweight="bold")
    ax.annotate(f"{dog[3]:.2f}", (scale[3], dog[3]), xytext=(2, -12),
                textcoords="offset points", color=ORANGE, fontsize=7.6, fontweight="bold")
    _panel(ax, "a", x=-0.25)


def panel_ld_decay(ax, empirical_ld):
    radius = np.asarray(empirical_ld["distance_mid_bp"], float)
    dog = np.asarray(empirical_ld["dog_mean_r2"], float)
    dog_low = np.asarray(empirical_ld["dog_ci95_low"], float)
    dog_high = np.asarray(empirical_ld["dog_ci95_high"], float)
    wolf = np.asarray(empirical_ld["wolf_r2"], float)
    ax.fill_between(radius, dog_low, dog_high, color=ORANGE, alpha=0.18, lw=0)
    ax.plot(radius, dog, "s-", color=ORANGE, lw=2.0, ms=4.4,
            label="dogs (33 sampled)")
    ax.plot(radius, wolf, "o-", color=BLUE, lw=2.0, ms=4.7,
            label="wolves ($n=33$)")
    ax.set_xscale("log")
    ax.set_xlim(250, 2_000_000)
    ax.set_ylim(0, 0.36)
    ax.set_xlabel("inter-SNP distance (bp)")
    ax.set_ylabel("mean $r^2$")
    ax.set_title("Dog LD is stronger at short distances", loc="left", fontsize=9.0)
    ax.legend(fontsize=7.2, loc="upper right", handlelength=1.4)
    _panel(ax, "b", x=-0.23)


def panel_canid_map(ax, dog):
    centers = np.asarray(dog["b_centers"], float)
    truth = np.asarray(dog["b_truth"], float)
    village = np.asarray(dog["b_vil"], float)
    breed = np.asarray(dog["b_brd"], float)
    ax.plot(centers, truth, color=BLACK, lw=2.4, label="exact shared map", zorder=4)
    ax.plot(centers, village, color=BLUE, lw=2.0, label="large-$N_e$ source", zorder=3)
    ax.plot(centers, breed, color=ORANGE, lw=1.8, label="breed, own LD", zorder=2)
    ax.set_yscale("log")
    positive = np.concatenate((truth, village, breed))
    positive = positive[positive > 0]
    ax.set_ylim(10 ** np.floor(np.log10(positive.min())),
                10 ** np.ceil(np.log10(positive.max())) * 1.25)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("position along shared region (Mb)")
    ax.set_ylabel("recombination rate (bp$^{-1}$)")
    ax.set_title("The bottleneck erases local map shape", loc="left", fontsize=9.4)
    ax.legend(fontsize=7.5, loc="lower left", handlelength=1.8)
    ax.text(0.98, 0.96,
            f"source vs exact  $r={_pearson_log(village, truth):.2f}$\n"
            f"breed vs exact   $r={_pearson_log(breed, truth):.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
            color="#333")
    _panel(ax, "c", x=-0.17)


def panel_paired_rescue(ax, dog):
    own = np.asarray(dog["own"], float)
    transfer = np.asarray(dog["trn"], float)
    keep = np.isfinite(own) & np.isfinite(transfer)
    own = own[keep]
    transfer = transfer[keep]
    improved = transfer >= own
    lim = (-0.32, 1.02)
    ax.plot(lim, lim, color="#888", lw=1.0, ls=(0, (3, 3)), zorder=0)
    ax.scatter(own[~improved], transfer[~improved], s=19, color=ORANGE,
               alpha=0.72, edgecolor="white", linewidth=0.25, zorder=2)
    ax.scatter(own[improved], transfer[improved], s=19, color=BLUE,
               alpha=0.62, edgecolor="white", linewidth=0.25, zorder=2)
    med_own = float(np.median(own))
    med_transfer = float(np.median(transfer))
    ax.scatter(med_own, med_transfer, s=92, color=BLACK, marker="D",
               edgecolor="white", linewidth=0.8, zorder=4)
    ax.annotate(f"median {med_own:.2f} $\u2192$ {med_transfer:.2f}",
                xy=(med_own, med_transfer), xytext=(-7, -25),
                textcoords="offset points", ha="center", va="top",
                fontsize=8.0, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLACK, lw=0.8))
    ax.text(0.96, 0.06, f"{improved.mean():.0%} improve\n$n={own.size}$ shared maps",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.7, color="#444")
    ax.text(0.94, 0.90, "source is better", transform=ax.transAxes,
            color=BLUE, fontsize=7.7, ha="right", va="top")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("breed, own LD ($r$)")
    ax.set_ylabel("source transfer ($r$)")
    ax.set_title("A larger population rescues the shared map", loc="left", fontsize=9.4)
    _panel(ax, "d", x=-0.23)


def panel_canid_severity(ax, dog):
    ne = np.asarray(dog["Ne"], float)
    own = np.asarray(dog["own"], float)
    transfer = np.asarray(dog["trn"], float)
    bx = np.asarray(dog["d_bx"], float)
    b_own = np.asarray(dog["d_own"], float)
    b_transfer = np.asarray(dog["d_trn"], float)
    ax.scatter(ne, own, s=10, color=ORANGE, alpha=0.18, lw=0, zorder=1)
    ax.scatter(ne, transfer, s=10, color=BLUE, alpha=0.18, lw=0, zorder=1)
    ax.fill_between(bx, b_own, b_transfer, color=BLUE, alpha=0.11, lw=0, zorder=2)
    ax.plot(bx, b_transfer, "o-", color=BLUE, lw=2.1, ms=5.0,
            label="larger source population", zorder=4)
    ax.plot(bx, b_own, "s-", color=ORANGE, lw=2.0, ms=4.8,
            label="bottlenecked population", zorder=3)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlim(520, 38)
    ax.set_ylim(-0.30, 1.02)
    ax.xaxis.set_major_locator(mticker.FixedLocator([50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["50", "100", "200", "500"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.axhline(0, color="#999", lw=0.7, zorder=0)
    ax.set_xlabel(r"present $N_e$  (deeper bottleneck $\rightarrow$)")
    ax.set_ylabel("100-kb map recovery ($r$)")
    ax.set_title("Rescue grows as the bottleneck deepens", loc="left", fontsize=9.4)
    ax.legend(fontsize=7.3, loc="lower right", handlelength=1.4)
    _panel(ax, "d", x=-0.25)


def panel_selfer_tracks(fig, cell, selfer):
    sub = cell.subgridspec(1, 5, wspace=0.12)
    axes = []
    false_hotspot = None
    for index, chrom in enumerate(CHROMS):
        ax = fig.add_subplot(sub[index])
        axes.append(ax)
        centers = np.asarray(selfer[f"c{chrom}_centers"], float)
        truth = np.asarray(selfer[f"c{chrom}_truth"], float)
        aware = np.asarray(selfer[f"c{chrom}_pred"], float)
        panmictic = np.asarray(selfer[f"c{chrom}_pyrho"], float)
        mask = (np.isfinite(truth) & (truth > 0) & np.isfinite(aware) & (aware > 0)
                & np.isfinite(panmictic) & (panmictic > 0))
        ztruth = _zmask(truth, mask)
        truth_show = _smooth(ztruth)
        aware_show = _smooth(_zmask(aware, mask))
        pan_show = _smooth(_zmask(panmictic, mask))
        lo, hi = _pericentromere(centers, ztruth, mask)
        if lo is not None:
            ax.axvline(centers[lo], color=BOUNDARY, lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.axvline(centers[hi], color=BOUNDARY, lw=0.7, ls=(0, (3, 2)), zorder=1)
        ax.axhline(0, color="#bbb", lw=0.65, zorder=1)
        ax.plot(centers, truth_show, color=BLACK, lw=1.8, zorder=4)
        ax.plot(centers, aware_show, color=BLUE, lw=1.45, zorder=3)
        ax.plot(
            centers,
            pan_show,
            color=GREEN,
            lw=1.35,
            ls=(0, (3, 2)),
            zorder=2,
        )
        aware_r = float(selfer[f"c{chrom}_r"])
        pan_r = float(selfer[f"c{chrom}_pyrho_r"])
        ax.set_title(f"chromosome {chrom}", fontsize=9.3, pad=8)
        ax.text(0.035, 0.96, f"{aware_r:+.2f}", color=BLUE,
                transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="top")
        ax.text(0.035, 0.84, f"{pan_r:+.2f}", color=GREEN,
                transform=ax.transAxes, fontsize=8.2, fontweight="bold", ha="left", va="top")
        if index == 0 and lo is not None:
            segment = np.arange(lo, hi + 1)
            peak = int(segment[np.nanargmax(pan_show[segment])])
            false_hotspot = (ax, centers[peak], pan_show[peak])
        ax.set_ylim(-2.8, 4.5)
        ax.set_xlabel("position (Mb)", fontsize=8.7)
        if index == 0:
            ax.set_ylabel("standardized rate ($z$)")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8.1)
    if false_hotspot is not None:
        ax, x, y = false_hotspot
        ax.annotate("false hotspot\nin cold pericentromere", xy=(x, min(y, 4.2)),
                    xytext=(0.56, 0.92), textcoords="axes fraction", fontsize=6.9,
                    color=GREEN, ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.9))
    _panel(axes[0], "e", x=-0.33, y=1.19)
    return axes


def panel_selfer_validation(ax, selfer, ceiling):
    x = np.arange(5)
    salome = np.array([float(selfer[f"c{c}_r"]) for c in CHROMS])
    rowan = np.asarray(ceiling["fastrho_real_recovery"]["per_chrom_vs_rowan"], float)
    panmictic = np.array([float(selfer[f"c{c}_pyrho_r"]) for c in CHROMS])
    ax.axhline(0, color="#666", lw=0.85)
    for xi, low, s, r in zip(x, panmictic, salome, rowan):
        ax.plot([xi, xi], [low, max(s, r)], color="#c0c0c0", lw=1.2, zorder=1)
    ax.scatter(x - 0.10, salome, s=48, color=LIGHT_BLUE, marker="o",
               edgecolor=BLACK, linewidth=0.45, label="selfing-aware vs Salomé", zorder=3)
    ax.scatter(x + 0.10, rowan, s=49, color=BLUE, marker="D",
               edgecolor=BLACK, linewidth=0.45, label="selfing-aware vs Rowan", zorder=3)
    ax.scatter(x, panmictic, s=48, color=GREEN, marker="s",
               edgecolor=BLACK, linewidth=0.45, label="panmictic vs Salomé", zorder=3)
    ax.axhline(rowan.mean(), color=BLUE, ls=(0, (4, 3)), lw=0.9)
    ax.axhline(panmictic.mean(), color=GREEN, ls=(0, (4, 3)), lw=0.9)
    ax.text(4.42, rowan.mean(), f"mean {rowan.mean():+.2f}", color=BLUE,
            fontsize=7.6, ha="right", va="bottom")
    ax.text(4.42, panmictic.mean(), f"mean {panmictic.mean():+.2f}", color=GREEN,
            fontsize=7.6, ha="right", va="top")
    ax.annotate("Salomé", (x[0] - 0.10, salome[0]), xytext=(-9, 10),
                textcoords="offset points", color="#5b8ea7", fontsize=7.3,
                ha="right", va="bottom")
    ax.annotate("Rowan", (x[0] + 0.10, rowan[0]), xytext=(9, 10),
                textcoords="offset points", color=BLUE, fontsize=7.3,
                ha="left", va="bottom")
    ax.annotate("panmictic", (x[0], panmictic[0]), xytext=(7, -10),
                textcoords="offset points", color=GREEN, fontsize=7.3,
                ha="left", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels([f"chr {c}" for c in CHROMS])
    ax.set_xlim(-0.46, 4.46)
    ax.set_ylim(-0.43, 0.47)
    ax.set_ylabel("Pearson $r$ with meiotic map")
    ax.set_title("Selfing against meiotic maps is positive",
                 loc="left", fontsize=9.4)
    _panel(ax, "f", x=-0.13)


def panel_selfer_ceiling(ax, ceiling):
    rows = (
        ("exact selfer map\n(clean simulation)",
         ceiling["clean_sim_recovery"]["selfer_self2"], BLACK, "o"),
        ("meiotic map\nagreement", ceiling["truth_map_ceiling_salome_vs_rowan"]["selfer_windows_100kb"],
         GRAY, "D"),
        ("selfing-aware\nvs Rowan", ceiling["fastrho_real_recovery"]["vs_rowan_mean"], BLUE, "o"),
        ("selfing-aware\nvs Salomé", ceiling["fastrho_real_recovery"]["vs_salome_mean"],
         LIGHT_BLUE, "o"),
        ("panmictic\nvs Salomé", ceiling["pyrho_real_recovery"]["vs_salome_mean"], GREEN, "s"),
    )
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#666", lw=0.85)
    ceiling_value = ceiling["truth_map_ceiling_salome_vs_rowan"]["selfer_windows_100kb"]
    ax.axvline(ceiling_value, color=GRAY, lw=0.9, ls=(0, (4, 3)))
    for yi, (_, value, color, marker) in zip(y, rows):
        ax.plot([0, value], [yi, yi], color=color, lw=2.0)
        ax.scatter(value, yi, s=55, color=color, marker=marker,
                   edgecolor=BLACK, linewidth=0.5, zorder=3)
        ax.annotate(f"{value:+.2f}", (value, yi),
                    xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                    ha="left" if value >= 0 else "right", va="center",
                    fontsize=7.9, fontweight="bold", color=color)
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=7.8)
    ax.set_xlim(-0.34, 1.0)
    ax.set_xlabel("100-kb map recovery (Pearson $r$)")
    ax.set_title("Reference mismatch reduces map recovery", loc="left", fontsize=9.4)
    ax.grid(axis="y", visible=False)
    _panel(ax, "g", x=-0.27)


def main():
    ps.style()
    plt.rcParams.update({"axes.titlesize": 9.7, "axes.labelsize": 9.2,
                         "xtick.labelsize": 8.2, "ytick.labelsize": 8.2})
    dog = np.load(os.path.join(FIGDATA, "dog_fig.npz"), allow_pickle=True)
    selfer = np.load(os.path.join(FIGDATA, "selfer_chroms.npz"), allow_pickle=True)
    with open(os.path.join(FIGDATA, "canid_empirical_scale.json")) as handle:
        empirical = json.load(handle)
    with open(os.path.join(FIGDATA, "canid_empirical_ld.json")) as handle:
        empirical_ld = json.load(handle)
    with open(os.path.join(FIGDATA, "selfer_ceiling.json")) as handle:
        ceiling = json.load(handle)

    fig = plt.figure(figsize=(11.2, 8.9))
    outer = fig.add_gridspec(4, 1, height_ratios=(1.65, 0.11, 1.82, 1.45),
                             hspace=0.42, left=0.055, right=0.985,
                             bottom=0.065, top=0.975)
    top = outer[0].subgridspec(1, 4, width_ratios=(1.30, 1.16, 1.78, 1.18), wspace=0.49)
    panel_empirical_canids(fig.add_subplot(top[0]), empirical)
    panel_ld_decay(fig.add_subplot(top[1]), empirical_ld)
    panel_canid_map(fig.add_subplot(top[2]), dog)
    panel_paired_rescue(fig.add_subplot(top[3]), dog)

    map_key = [Line2D([0], [0], color=BLACK, lw=2.0, label="meiotic map"),
               Line2D([0], [0], color=BLUE, lw=2.0, label="selfing-aware"),
               Line2D([0], [0], color=GREEN, lw=2.0, ls=(0, (3, 2)),
                      label="panmictic"),
               Line2D([0], [0], color=BOUNDARY, lw=0.9, ls=(0, (3, 2)),
                      label="cold-pericentromere bounds")]
    key_ax = fig.add_subplot(outer[1])
    key_ax.axis("off")
    key_ax.legend(handles=map_key, loc="center", ncol=4, frameon=False,
                  fontsize=7.5, handlelength=1.7, columnspacing=1.35)
    panel_selfer_tracks(fig, outer[2], selfer)
    bottom = outer[3].subgridspec(1, 2, width_ratios=(1.35, 1.0), wspace=0.46)
    panel_selfer_validation(fig.add_subplot(bottom[0]), selfer, ceiling)
    panel_selfer_ceiling(fig.add_subplot(bottom[1]), ceiling)

    os.makedirs(OUT, exist_ok=True)
    pdf = os.path.join(OUT, "fig2_history_rescue.pdf")
    png = os.path.join(OUT, "fig2_history_rescue.png")
    fig.savefig(pdf, dpi=600, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    print(pdf)


if __name__ == "__main__":
    main()
