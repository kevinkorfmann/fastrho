"""Manuscript Figure 5: arrangement-stratified redpoll supergene analysis.

This cache-only renderer separates the evidence into four compact panels: the
inversion-PCA karyotype assignments, aligned chromosome-1 maps, the long-range
LD endpoint, and estimator-independent LD decay.  The figure treats the
supergene as established and limits its headline claim to attenuation of the
pooled LD cold block after stratification.

No inference is performed here.  Every plotted quantity is read from committed
figure-data archives.
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paper_style as ps
from fastrho.preprocess import mean_rate_between


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "paper", "figdata")

BLUE = "#2737E7"
ORANGE = "#151515"
PURPLE = "#777777"
GREY = "#9A9A9A"
BLACK = "#151515"
BOUNDARY = "#777777"


def _load_data():
    fieldguide = np.load(os.path.join(DATA, "fieldguide_redpoll.npz"), allow_pickle=True)
    maps = np.load(os.path.join(DATA, "redpoll_karyotype_maps.npz"), allow_pickle=True)
    with open(os.path.join(DATA, "redpoll_karyotype_maps.json"), encoding="utf-8") as handle:
        map_stats = json.load(handle)
    with open(os.path.join(DATA, "redpoll_karyotype_null.json"), encoding="utf-8") as handle:
        null = json.load(handle)
    with open(os.path.join(DATA, "redpoll_karyotype_ld.json"), encoding="utf-8") as handle:
        ld = json.load(handle)
    return fieldguide, maps, map_stats, null, ld


def _pooled_map(fieldguide, maps):
    bp = np.r_[fieldguide["pos_left"][0], fieldguide["pos_right"]]
    return mean_rate_between(bp, fieldguide["rho_per_bp"], maps["edges"])


def _ratio(x, inside, flank):
    return float(np.nanmedian(x[inside]) / np.nanmedian(x[flank]))


def _smooth(x, width=4):
    x = np.asarray(x, float)
    valid = np.isfinite(x).astype(float)
    filled = np.where(np.isfinite(x), x, 0.0)
    kernel = np.ones(width, float)
    numerator = np.convolve(filled, kernel, mode="same")
    denominator = np.convolve(valid, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def _map_key(fig, cell):
    ax = fig.add_subplot(cell)
    ax.axis("off")
    key = [
        Line2D([0], [0], color=GREY, lw=2.2, label="pooled"),
        Line2D([0], [0], color=BLUE, lw=2.2, label="A/A"),
        Line2D([0], [0], color=PURPLE, lw=0, marker="o", ms=5.0, label="A/B"),
        Line2D([0], [0], color=ORANGE, lw=2.2, label="B/B"),
    ]
    ax.legend(handles=key, loc="center", ncol=4, frameon=False,
              fontsize=7.5, handlelength=1.45, columnspacing=1.1)
    return ax


def _panel_pca(ax, fieldguide):
    labels = np.asarray(fieldguide["pca_labels"], int)
    pc1 = np.asarray(fieldguide["pca_pc1"], float)
    pc2 = np.asarray(fieldguide["pca_pc2"], float)
    sizes = np.bincount(labels, minlength=3)
    homo = [int(x) for x in np.argsort(sizes)[-2:]]
    homo.sort(key=lambda x: float(np.mean(pc1[labels == x])))
    hetero = int(({0, 1, 2} - set(homo)).pop())
    groups = ((homo[0], "A/A", BLUE, "o"), (hetero, "A/B", PURPLE, "D"),
              (homo[1], "B/B", ORANGE, "s"))
    for label, name, color, marker in groups:
        keep = labels == label
        ax.scatter(pc1[keep], pc2[keep], s=24, color=color, marker=marker,
                   alpha=0.82, edgecolor="white", linewidth=0.45, zorder=3)
    for y, (_, name, color, _marker) in zip((0.96, 0.87, 0.78), groups):
        label = {"A/A": homo[0], "A/B": hetero, "B/B": homo[1]}[name]
        ax.text(0.035, y, f"{name}  $n={int((labels == label).sum())}$",
                transform=ax.transAxes, color=color, fontsize=7.4,
                fontweight="bold", ha="left", va="top")
    ax.axvline(0, color="#bbb", lw=0.65, zorder=0)
    ax.axhline(0, color="#bbb", lw=0.65, zorder=0)
    ax.set_xlabel(f"inversion PC1 ({100 * float(fieldguide['pca_ev'][0]):.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * float(fieldguide['pca_ev'][1]):.1f}%)")
    ax.set_title("Three inferred karyotypes", loc="left", fontsize=8.4)
    ps.panel(ax, "a", x=-0.23, y=1.10, fontsize=13)


def _panel_maps(fig, cell, centers, maps_by_group, inv0, inv1):
    host = fig.add_subplot(cell)
    host.axis("off")
    host.set_title(
        "Stratification reduces map trough of pooled maps",
        loc="left",
        x=0.06,
        fontsize=8.4,
        pad=7,
    )
    ps.panel(host, "b", x=-0.065, y=1.11, fontsize=13)
    sub = cell.subgridspec(3, 1, hspace=0.08)
    rows = (("pooled  $n=72$", GREY, maps_by_group["pooled"]),
            ("A/A  $n=37$", BLUE, maps_by_group["A"]),
            ("B/B  $n=28$", ORANGE, maps_by_group["B"]))
    positive = np.concatenate([np.asarray(row[2], float) for row in rows])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    lo = 10 ** np.floor(np.log10(np.nanpercentile(positive, 0.5)))
    hi = 10 ** np.ceil(np.log10(np.nanpercentile(positive, 99.5)))
    axes = []
    for index, (label, color, values) in enumerate(rows):
        ax = fig.add_subplot(sub[index], sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(centers, values, color=color, lw=0.65, alpha=0.25, zorder=1)
        ax.plot(centers, _smooth(values), color=color, lw=1.85, zorder=2)
        ax.axvline(inv0, color=BOUNDARY, lw=0.65, ls=(0, (3, 2)))
        ax.axvline(inv1, color=BOUNDARY, lw=0.65, ls=(0, (3, 2)))
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
        ax.tick_params(labelleft=index == 1)
        ax.text(
            0.012,
            0.66,
            label,
            transform=ax.transAxes,
            color=color,
            fontsize=7.8,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=0.9),
            zorder=10,
        )
        if index == 0:
            ax.text(
                inv1 - 1.0,
                0.97,
                "known supergene 18.9–75.0 Mb",
                transform=ax.get_xaxis_transform(),
                color=BLACK,
                fontsize=7.0,
                ha="right",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=0.9),
                zorder=10,
            )
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("position on chromosome 1 (Mb)")
    axes[1].set_ylabel(r"population-scaled rate, $\hat\rho$")
    return axes


def _ld_arrays(ld, key):
    group = ld["groups"][key]
    x = np.array([row["distance_mid"] for row in group["inside"]], float) / 1e3
    inside = np.array([row["mean_r2_corrected"] for row in group["inside"]], float)
    flanks = np.array([row["mean_r2_corrected"] for row in group["flanks"]], float)
    return x, inside, flanks


def _panel_summary(fig, cell, ratios, null, ld):
    """Pair the decisive matched-size control with raw-LD corroboration."""

    section = cell.subgridspec(3, 1, height_ratios=(0.16, 1.08, 0.92), hspace=0.50)
    header = fig.add_subplot(section[0])
    header.axis("off")
    header.text(
        0,
        0.65,
        "Equal size does not reduce the suppression",
        transform=header.transAxes,
        fontsize=8.4,
        ha="left",
        va="center",
    )
    ps.panel(header, "c", x=-0.16, y=1.05, fontsize=13)

    ax = fig.add_subplot(section[1])
    colors = (GREY, BLUE, ORANGE)
    labels = ("pooled", "A/A", "B/B")
    values = np.asarray((ratios["pooled"], ratios["A"], ratios["B"]), float)
    null_by_size = {
        1: np.asarray(
            [
                row["inside_flank_ratio"]
                for row in null["records"]
                if row["size"] == 37
            ],
            float,
        ),
        2: np.asarray(
            [
                row["inside_flank_ratio"]
                for row in null["records"]
                if row["size"] == 28
            ],
            float,
        ),
    }
    rng = np.random.default_rng(17)
    for index, (value, color) in enumerate(zip(values, colors)):
        ax.plot([index, index], [0, value], color=color, lw=1.8, alpha=0.85)
        ax.scatter(
            index,
            value,
            s=45,
            color=color,
            edgecolor=BLACK,
            linewidth=0.5,
            zorder=4,
        )
        ax.text(
            index,
            value + 0.045,
            f"{value:.2f}",
            color=color,
            fontsize=7.1,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    for index, null_values in null_by_size.items():
        jitter = rng.uniform(-0.13, 0.13, null_values.size)
        ax.scatter(
            index + jitter,
            null_values,
            s=19,
            facecolor="white",
            edgecolor="#555",
            linewidth=0.65,
            zorder=3,
        )
    ax.axhline(1, color="#888", lw=0.7, ls=(0, (4, 3)))
    ax.set_xticks(range(3), labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Rate inside / flanks")
    ax.set_title("Map comparison", loc="left", fontsize=8.4, pad=4)

    ax = fig.add_subplot(section[2])
    rows = (("pooled", GREY, "pooled"), ("A/A", BLUE, "arrangement_A"),
            ("B/B", ORANGE, "arrangement_B"))
    y_positions = np.arange(len(rows))[::-1]
    for y, (_label, color, key) in zip(y_positions, rows):
        _, inside_ld, flanks_ld = _ld_arrays(ld, key)
        yi, yf = float(inside_ld[-1]), float(flanks_ld[-1])
        fold = yi / yf
        fold_label = f"{fold:.0f}× flank" if fold >= 10 else f"{fold:.2f}× flank"
        ax.plot([yf, yi], [y, y], color=color, lw=2.1, alpha=0.78,
                solid_capstyle="round", zorder=1)
        ax.scatter(yi, y, s=62, color=color, edgecolor=BLACK,
                   linewidth=0.55, zorder=3)
        ax.scatter(yf, y, s=50, marker="D", facecolor="white", edgecolor=color,
                   linewidth=1.25, zorder=3)
        ax.annotate(
            fold_label,
            xy=(max(yi, yf), y),
            xytext=(7, 0),
            textcoords="offset points",
            color=color,
            fontsize=7.0,
            fontweight="bold",
            ha="left",
            va="center",
        )
    ax.set_xscale("log")
    ax.set_xlim(5.5e-4, 7.5e-2)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y_positions, [row[0] for row in rows])
    for tick, (_, color, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_xlabel(r"Corrected $r^2$ at 250--500 kb")
    ax.set_title(
        r"LD endpoint ($\bullet$ interior; $\diamond$ flanks)",
        loc="left",
        fontsize=8.0,
        pad=4,
    )
    ax.grid(axis="x", which="major", color="#e8e8e8", lw=0.65)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _panel_ld_decay(fig, cell, ld):
    host = fig.add_subplot(cell)
    host.axis("off")
    host.set_title("Dosage LD declines toward flank levels within homokaryotypes",
                   loc="left", fontsize=9.5, pad=7)
    ps.panel(host, "d", x=-0.055, y=1.12, fontsize=13)
    sub = cell.subgridspec(1, 3, wspace=0.13)
    rows = (("pooled", GREY, "pooled"), ("A/A", BLUE, "arrangement_A"),
            ("B/B", ORANGE, "arrangement_B"))
    axes = []
    for index, (title, color, key) in enumerate(rows):
        ax = fig.add_subplot(sub[index], sharex=axes[0] if axes else None,
                             sharey=axes[0] if axes else None)
        axes.append(ax)
        x, inside, flanks = _ld_arrays(ld, key)
        ax.plot(x, inside, color=color, lw=1.75, label="inside inversion")
        ax.plot(x, flanks, color=color, lw=1.25, ls=(0, (3, 2)),
                alpha=0.75, label="flanks")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(7e-4, 8e-2)
        ax.text(0.06, 0.95, title, transform=ax.transAxes, color=color,
                fontsize=8.3, fontweight="bold", ha="left", va="top")
        if index == 0:
            ax.set_ylabel(r"corrected dosage $r^2$")
            ax.legend(loc="lower left", fontsize=6.8, handlelength=1.35)
        else:
            ax.tick_params(labelleft=False)
        if index == 1:
            ax.set_xlabel("SNP separation (kb)")
    return axes


def main():
    ps.style()
    plt.rcParams.update({"axes.titlesize": 9.5, "axes.labelsize": 8.8,
                         "xtick.labelsize": 7.8, "ytick.labelsize": 7.8})
    fieldguide, maps, _map_stats, null, ld = _load_data()
    centers = np.asarray(maps["centers"], float) / 1e6
    inv0 = float(maps["inv_start"]) / 1e6
    inv1 = float(maps["inv_end"]) / 1e6
    maps_by_group = {"pooled": _pooled_map(fieldguide, maps),
                     "A": np.asarray(maps["arrangement_A_rate"], float),
                     "B": np.asarray(maps["arrangement_B_rate"], float)}
    inside = (centers >= inv0) & (centers < inv1)
    flank = ~inside
    ratios = {
        key: _ratio(value, inside, flank)
        for key, value in maps_by_group.items()
    }

    fig = plt.figure(figsize=(7.15, 3.65))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.09, 1.0),
        hspace=0.12,
        left=0.055,
        right=0.985,
        bottom=0.10,
        top=0.975,
    )
    _map_key(fig, outer[0])
    panels = outer[1].subgridspec(
        1,
        3,
        width_ratios=(0.92, 2.18, 1.16),
        wspace=0.40,
    )
    _panel_pca(fig.add_subplot(panels[0]), fieldguide)
    _panel_maps(fig, panels[1], centers, maps_by_group, inv0, inv1)
    _panel_summary(fig, panels[2], ratios, null, ld)

    ps.save(fig, "fig_redpoll_karyotype", formats=("pdf", "png"), dpi=600)


if __name__ == "__main__":
    main()
