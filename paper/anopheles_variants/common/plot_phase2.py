#!/usr/bin/env python3
"""Generate publication figures for the Phase 2 manuscript variant.

The layouts present the evidence available in the open Phase 2 release.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import paper_style as ps  # noqa: E402

ARMS = ("2R", "2L", "3R", "3L", "X")
AUTOSOMES = ARMS[:-1]
ARM_MARKERS = {"2R": "o", "2L": "s", "3R": "^", "3L": "D"}
BLUE = ps.C["fastrho"]
INK = ps.C["truth"]
GRAY = ps.C["relernn"]
LIGHT_GRAY = "#E7E7E4"
SPECIES_COLORS = {
    "Anopheles gambiae": BLUE,
    "Anopheles coluzzii": INK,
}
SPECIES_LABELS = {
    "Anopheles gambiae": r"$A.$ gambiae",
    "Anopheles coluzzii": r"$A.$ coluzzii",
}
BREAKPOINTS = (20.524, 42.166)


def style() -> None:
    ps.style()
    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
        }
    )


def panel(ax: mpl.axes.Axes, label: str, *, x: float = -0.11, y: float = 1.11) -> None:
    ps.panel(ax, label.lower(), x=x, y=y, fontsize=9)


def save(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_map(path: Path, window: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return (
            archive[f"starts_{window}"].astype(float),
            archive[f"r_{window}"].astype(float),
        )


def smooth_rows(matrix: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.empty_like(matrix, dtype=float)
    for index, row in enumerate(np.asarray(matrix, float)):
        valid = np.isfinite(row).astype(float)
        numerator = np.convolve(np.nan_to_num(row, nan=0.0), kernel, mode="same")
        denominator = np.convolve(valid, kernel, mode="same")
        smoothed[index] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )
    return smoothed


def bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(
        rng.choice(array, size=(4_000, len(array)), replace=True),
        axis=1,
    )
    return tuple(np.percentile(draws, [2.5, 97.5]))


def rankdata(values: list[float] | np.ndarray) -> np.ndarray:
    """Average ranks with tie handling, without importing the heavy SciPy stack."""
    array = np.asarray(values, dtype=float)
    _unique, inverse, counts = np.unique(array, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    average_ranks = (cumulative - counts + 1 + cumulative) / 2
    return average_ranks[inverse]


def spearman(values_a: list[float] | np.ndarray, values_b: list[float] | np.ndarray) -> float:
    return float(np.corrcoef(rankdata(values_a), rankdata(values_b))[0, 1])


def locus_label(locus: str) -> str:
    return locus.replace("Cyp6aa/Cyp6p", "Cyp6aa/p").replace("D7r2/D7r4", "D7r2/r4")


def atlas_figure(
    maps: Path,
    selection: list[dict[str, str]],
    inversion: dict,
    resistance: dict,
    out: Path,
) -> None:
    primary = resistance["panels"]["hancock_mechanisms"]
    species_order = ("Anopheles gambiae", "Anopheles coluzzii")

    map_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    arm_lengths: dict[str, float] = {}
    for row in selection:
        cohort = row["cohort"]
        map_data[cohort] = {}
        for arm in ARMS:
            starts, rates = load_map(maps / f"{cohort}__{arm}.npz")
            positive = rates[(rates > 0) & np.isfinite(rates)]
            normalized = rates / np.nanmedian(positive)
            positions = (starts + 25_000) / 1e6
            map_data[cohort][arm] = (positions, normalized)
            arm_lengths[arm] = max(arm_lengths.get(arm, 0), float(positions[-1] + 0.025))

    offsets: dict[str, float] = {}
    cursor = 0.0
    for arm in ARMS:
        offsets[arm] = cursor
        cursor += arm_lengths[arm]
    total = cursor

    fig = plt.figure(figsize=(7.15, 7.55))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=(1.56, 1.18, 1.43, 1.00),
        hspace=0.60,
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.985,
    )

    # a | Genome-wide context.
    atlas_grid = outer[0].subgridspec(3, 1, height_ratios=(0.62, 1, 1), hspace=0.07)
    label_axis = fig.add_subplot(atlas_grid[0])
    rate_axes = [fig.add_subplot(atlas_grid[index]) for index in (1, 2)]
    label_axis.set_xlim(0, total)
    label_axis.set_ylim(0, 1)
    label_axis.set_axis_off()
    label_axis.set_title("Recombination maps for five chromosome arms", loc="left", y=1.08, pad=3)
    panel(label_axis, "a")

    species_medians: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for axis_index, (species, ax) in enumerate(zip(species_order, rate_axes, strict=True)):
        cohorts = [row["cohort"] for row in selection if row["species"] == species]
        color = SPECIES_COLORS[species]
        for cohort in cohorts:
            for arm in ARMS:
                position, rates = map_data[cohort][arm]
                ax.plot(offsets[arm] + position, rates, color=color, alpha=0.17, lw=0.32)
        for arm in ARMS:
            arrays = [map_data[cohort][arm][1] for cohort in cohorts]
            length = min(map(len, arrays))
            position = map_data[cohorts[0]][arm][0][:length]
            median = np.nanmedian(np.asarray([array[:length] for array in arrays]), axis=0)
            species_medians[(species, arm)] = (position, median)
            ax.plot(offsets[arm] + position, median, color=color, lw=0.95)
        ax.set_ylabel(SPECIES_LABELS[species], color=color, fontsize=6.0, labelpad=4)
        ax.set_xlim(0, total)
        ax.set_ylim(0, 12)
        ax.grid(axis="y", color="0.92", lw=0.45)
        ax.tick_params(axis="x", labelbottom=axis_index == len(rate_axes) - 1)

    for ax in rate_axes:
        for boundary in (offsets["2L"] + BREAKPOINTS[0], offsets["2L"] + BREAKPOINTS[1]):
            ax.axvline(boundary, color="0.58", lw=0.55, ls=(0, (2.2, 2.2)), zorder=-1)
        for arm in ARMS[1:]:
            ax.axvline(offsets[arm], color="0.84", lw=0.5, zorder=-1)

    definition_by_locus = {
        locus: primary["rows"][0]["loci"][locus]["target"] for locus in primary["loci"]
    }
    label_transform = mtransforms.blended_transform_factory(label_axis.transData, label_axis.transAxes)
    lane_right = [-np.inf, -np.inf, -np.inf]
    lane_y = (0.08, 0.43, 0.78)
    ordered_loci = sorted(
        definition_by_locus.items(),
        key=lambda item: offsets[item[1]["arm"]] + float(item[1]["mb"]),
    )
    for locus, record in ordered_loci:
        arm = record["arm"]
        mb = float(record["mb"])
        genome_x = offsets[arm] + mb
        for species, ax in zip(species_order, rate_axes, strict=True):
            position, median = species_medians[(species, arm)]
            local = np.abs(position - mb) <= 0.15
            local_rate = (
                float(np.nanmedian(median[local]))
                if np.any(local)
                else float(median[int(np.nanargmin(np.abs(position - mb)))])
            )
            ax.axvline(genome_x, color=INK, alpha=0.20, lw=0.5, zorder=-1)
            ax.scatter(
                genome_x,
                min(local_rate, 11.65),
                marker="^",
                s=24,
                facecolor=INK,
                edgecolor="white",
                linewidth=0.55,
                zorder=6,
                clip_on=False,
            )
        short = locus_label(locus)
        half_width = max(2.5, 0.58 * len(short))
        label_x = float(np.clip(genome_x, half_width, total - half_width))
        lane = next(
            (
                index
                for index, previous_right in enumerate(lane_right)
                if label_x - half_width >= previous_right + 1.0
            ),
            int(np.argmin(lane_right)),
        )
        lane_right[lane] = label_x + half_width
        label_axis.plot(
            [genome_x, label_x],
            [0, lane_y[lane] - 0.05],
            color=INK,
            alpha=0.30,
            lw=0.4,
            clip_on=False,
        )
        label_axis.text(
            label_x,
            lane_y[lane],
            short,
            transform=label_transform,
            ha="center",
            va="bottom",
            color=INK,
            fontsize=4.8,
            fontweight="bold",
            clip_on=False,
        )

    centers = [offsets[arm] + arm_lengths[arm] / 2 for arm in ARMS]
    rate_axes[-1].set_xticks(centers, ARMS)
    rate_axes[-1].set_xlabel("Chromosome arm")
    rate_axes[0].text(offsets["2L"] + np.mean(BREAKPOINTS), 11.3, "2La", ha="center", color=INK)

    # b-c | 2La landmark analysis.
    validation_grid = outer[1].subgridspec(1, 2, width_ratios=(1.34, 1.0), wspace=0.47)
    validation_rows = sorted(inversion["rows"], key=lambda row: row["het_expected"])
    ax_heat = fig.add_subplot(validation_grid[0, 0])
    heat_sources = []
    for row in validation_rows:
        row_positions, rates = map_data[row["pop"]]["2L"]
        outside = (row_positions < BREAKPOINTS[0]) | (row_positions > BREAKPOINTS[1])
        heat_sources.append(
            (row_positions, np.log2(np.maximum(rates / np.nanmedian(rates[outside]), 2**-3)))
        )
    shared_start = max(source[0][0] for source in heat_sources)
    shared_end = min(source[0][-1] for source in heat_sources)
    positions = np.arange(np.ceil(shared_start * 20) / 20, np.floor(shared_end * 20) / 20 + 0.025, 0.05)
    heat = [np.interp(positions, row_positions, values) for row_positions, values in heat_sources]
    matrix = np.asarray(heat)
    image = ax_heat.imshow(
        smooth_rows(matrix),
        aspect="auto",
        interpolation="nearest",
        extent=(positions[0], positions[-1], len(validation_rows) - 0.5, -0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2),
    )
    for boundary in BREAKPOINTS:
        ax_heat.axvline(boundary, color="black", lw=0.65)
    ax_heat.set_yticks(
        np.arange(len(validation_rows)),
        [f"{row['pop']}  H={row['het_expected']:.2f}" for row in validation_rows],
    )
    for tick, row in zip(ax_heat.get_yticklabels(), validation_rows, strict=True):
        tick.set_color(SPECIES_COLORS[row["taxon"]])
    ax_heat.set_xlabel("Chromosome 2L position (Mb)")
    ax_heat.set_title("Cold block depth varies between cohorts", loc="left")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02)
    colorbar.set_label(r"$\log_2$ rate ratio")
    panel(ax_heat, "b")

    ax_prediction = fig.add_subplot(validation_grid[0, 1])
    for row in inversion["rows"]:
        ax_prediction.scatter(
            row["het_expected"],
            row["suppression_depth"],
            s=24,
            color=SPECIES_COLORS[row["taxon"]],
            edgecolor="white",
            linewidth=0.5,
        )
    h = np.asarray([row["het_expected"] for row in inversion["rows"]])
    depth = np.asarray([row["suppression_depth"] for row in inversion["rows"]])
    slope, intercept = np.polyfit(h, depth, 1)
    xx = np.linspace(0, 0.51, 100)
    ax_prediction.plot(xx, slope * xx + intercept, color="0.25", lw=1.0)
    ax_prediction.axhline(0, color="0.6", ls="--", lw=0.7)
    ax_prediction.set_xlim(-0.02, 0.52)
    ax_prediction.set_xlabel("Expected 2La heterokaryotype frequency")
    ax_prediction.set_ylabel(r"Suppression depth $1-r_{\rm in}/r_{\rm out}$")
    ax_prediction.text(
        0.03,
        0.97,
        "Pearson $r=%.2f$, $P=%.3f$\nSpearman $r_s=%.2f$, $P=%.3f$\n$n=%d$ cohorts"
        % (
            inversion["pearson_Hexp_depth"][0],
            inversion["pearson_Hexp_depth"][1],
            inversion["spearman_Hexp_depth"][0],
            inversion["spearman_Hexp_depth"][1],
            len(inversion["rows"]),
        ),
        transform=ax_prediction.transAxes,
        va="top",
        fontsize=6.2,
    )
    ax_prediction.set_title("Suppression versus expected arrangement mixing", loc="left")
    panel(ax_prediction, "c")

    # d | Locus-by-cohort effect matrix and its directional margin.
    ordered_rows = sorted(
        primary["rows"],
        key=lambda row: (0 if row["species"] == "Anopheles gambiae" else 1, row["cohort"]),
    )
    loci = list(primary["loci"])
    resistance_grid = outer[2].subgridspec(1, 3, width_ratios=(5.2, 0.18, 1.0), wspace=0.20)
    ax_resistance = fig.add_subplot(resistance_grid[0, 0])
    log2_ratios = np.asarray(
        [[np.log2(row["loci"][locus]["ratio"]) for row in ordered_rows] for locus in loci]
    )
    cold_warm = LinearSegmentedColormap.from_list("phase2_cold_equal_warm", [BLUE, "#FAFAF8", GRAY])
    resistance_image = ax_resistance.imshow(
        log2_ratios,
        aspect="auto",
        interpolation="nearest",
        cmap=cold_warm,
        norm=TwoSlopeNorm(vmin=-4, vcenter=0, vmax=2),
    )
    ax_resistance.set_xticks(
        np.arange(len(ordered_rows)),
        [row["cohort"] for row in ordered_rows],
        rotation=48,
        ha="right",
    )
    for tick, row in zip(ax_resistance.get_xticklabels(), ordered_rows, strict=True):
        tick.set_color(SPECIES_COLORS[row["species"]])
    ax_resistance.set_yticks(np.arange(len(loci)), [locus_label(locus) for locus in loci])
    ax_resistance.axvline(4.5, color="0.15", lw=1.0)
    ax_resistance.set_title("Resistance-region rates are lower than matched controls", loc="left")
    panel(ax_resistance, "d")

    color_axis = fig.add_subplot(resistance_grid[0, 1])
    colorbar = fig.colorbar(resistance_image, cax=color_axis)
    colorbar.set_ticks([-4, -2, 0, 2])
    colorbar.set_ticklabels(["≤−4", "−2", "0", "2"])
    color_axis.set_title(r"$\log_2$" "\nratio", fontsize=6.0, pad=3)

    count_axis = fig.add_subplot(resistance_grid[0, 2], sharey=ax_resistance)
    cold_counts = np.sum(log2_ratios < 0, axis=1)
    y = np.arange(len(loci))
    count_axis.barh(y, cold_counts, color=BLUE, height=0.64)
    for yi, count in zip(y, cold_counts, strict=True):
        inside = count >= 7
        count_axis.text(
            count - 0.18 if inside else count + 0.18,
            yi,
            f"{count}/9",
            ha="right" if inside else "left",
            va="center",
            fontsize=5.5,
            color="white" if inside else "black",
        )
    count_axis.set_xlim(0, 9)
    count_axis.set_xticks([0, 3, 6, 9])
    count_axis.tick_params(axis="y", left=False, labelleft=False)
    count_axis.set_xlabel("Cohorts with\nratio < 1")
    count_axis.set_title("Direction in cohorts", loc="left", fontsize=6.5)
    count_axis.grid(axis="x", color="0.90", lw=0.5)

    # e-f | Cohort heterogeneity and the 15-region species summary.
    controls_grid = outer[3].subgridspec(1, 2, width_ratios=(0.92, 1.48), wspace=0.52)
    ax_cohorts = fig.add_subplot(controls_grid[0, 0])
    cohort_rows = sorted(primary["rows"], key=lambda row: row["ratio"])
    y = np.arange(len(cohort_rows))
    for yi, row in zip(y, cohort_rows, strict=True):
        significant = row["perm_p"] < 0.05
        color = SPECIES_COLORS[row["species"]]
        ax_cohorts.plot(
            np.log2(row["ratio"]),
            yi,
            marker="o",
            ms=4.6,
            markerfacecolor=color if significant else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            ls="",
        )
    ax_cohorts.axvline(0, color="0.45", lw=0.8)
    ax_cohorts.set_yticks(y, [row["cohort"] for row in cohort_rows])
    for tick, row in zip(ax_cohorts.get_yticklabels(), cohort_rows, strict=True):
        tick.set_color(SPECIES_COLORS[row["species"]])
    ax_cohorts.set_xlabel(r"Cohort median $\log_2$(focal/control)")
    ax_cohorts.set_title("Population-level summary", loc="left")
    panel(ax_cohorts, "e", x=-0.16)

    ax_sensitivity = fig.add_subplot(controls_grid[0, 1])
    species_order = ("Anopheles gambiae", "Anopheles coluzzii")
    y_positions = np.arange(len(species_order) - 1, -1, -1, dtype=float)
    for index, species in enumerate(species_order):
        group = [float(row["ratio"]) for row in primary["rows"] if row["species"] == species]
        median = float(np.median(group))
        interval = bootstrap_median_ci(group, 1000 + index)
        offsets = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else np.zeros(1)
        ax_sensitivity.scatter(
            group,
            y_positions[index] + offsets,
            s=22,
            marker="o",
            facecolor="white",
            edgecolor=SPECIES_COLORS[species],
            linewidth=0.9,
            zorder=3,
        )
        ax_sensitivity.errorbar(
            median,
            y_positions[index],
            xerr=np.asarray([[median - interval[0]], [interval[1] - median]]),
            color=SPECIES_COLORS[species],
            marker="D",
            markerfacecolor=SPECIES_COLORS[species],
            markeredgecolor="white",
            markeredgewidth=0.45,
            ms=5.8,
            lw=1.5,
            capsize=3,
            zorder=4,
        )
        ax_sensitivity.text(
            median,
            y_positions[index] + 0.27,
            f"median {median:.2f}",
            color=SPECIES_COLORS[species],
            fontsize=6.2,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ax_sensitivity.axvline(1, color=INK, lw=0.9)
    ax_sensitivity.text(
        0.995,
        1.42,
        "equal rates",
        rotation=90,
        ha="right",
        va="top",
        color=INK,
        fontsize=5.6,
    )
    ax_sensitivity.set_yticks(
        y_positions,
        [r"$\it{A.\ gambiae}$  ($n=5$)", r"$\it{A.\ coluzzii}$  ($n=4$)"],
    )
    ax_sensitivity.set_xlim(0.38, 1.04)
    ax_sensitivity.set_ylim(-0.42, 1.52)
    ax_sensitivity.set_xlabel("Resistance-region rate / matched-control rate")
    ax_sensitivity.set_title("Resistance regions are colder in both species", loc="left")
    ax_sensitivity.spines["left"].set_visible(False)
    ax_sensitivity.tick_params(axis="y", length=0)
    panel(ax_sensitivity, "f", x=-0.40)

    save(fig, out / "fig_phase2_anopheles")


def workflow_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    edgecolor: str,
    facecolor: str = "white",
    fontsize: float = 6.2,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.15,
    )


def pedigree_shift_null(pedigree: dict) -> np.ndarray:
    arm_rows: dict[str, list[dict]] = {}
    direct = []
    for arm in AUTOSOMES:
        rows = sorted((row for row in pedigree["windows"] if row["arm"] == arm), key=lambda row: row["start"])
        arm_rows[arm] = rows
        direct.extend(float(row["direct_normalized"]) for row in rows if row["supported"] and row["direct_normalized"] is not None)
    null = []
    shift_ranges = [range(len(arm_rows[arm])) for arm in AUTOSOMES]
    for shifts in itertools.product(*shift_ranges):
        if not any(shifts):
            continue
        shifted = []
        for arm, shift in zip(AUTOSOMES, shifts, strict=True):
            rows = arm_rows[arm]
            consensus = np.asarray([row["atlas_normalized"] for row in rows], dtype=float)
            supported = np.asarray([row["supported"] and row["direct_normalized"] is not None for row in rows])
            shifted.extend(np.roll(consensus, shift)[supported])
        null.append(spearman(direct, shifted))
    return np.asarray(null)


def pedigree_figure(pedigree: dict, out: Path) -> None:
    mpl.rcParams.update(
        {
            "font.size": 7.5,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
        }
    )
    supported = [
        row
        for row in pedigree["windows"]
        if row["supported"] and row["direct_normalized"] is not None
    ]
    fig = plt.figure(figsize=(7.25, 3.45))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.55, 1.50, 1.40),
        left=0.04,
        right=0.985,
        bottom=0.18,
        top=0.87,
        wspace=0.40,
    )
    ax_a, ax_b, ax_c = (fig.add_subplot(grid[0, index]) for index in range(3))

    ax_a.set_axis_off()
    ax_a.set_title("Pedigree comparison setup", loc="left", pad=5)
    workflow_box(
        ax_a,
        (0.03, 0.69),
        0.41,
        0.23,
        f"Phase 2 crosses\n{pedigree['n_crosses']} families\n{pedigree['n_events_width_le_1mb']} crossovers",
        edgecolor=BLUE,
        facecolor="#F3F4FF",
        fontsize=7.0,
    )
    workflow_box(
        ax_a,
        (0.56, 0.69),
        0.41,
        0.23,
        "Inferred maps\n9 populations",
        edgecolor=INK,
        facecolor="#F5F5F3",
        fontsize=7.0,
    )
    workflow_box(
        ax_a,
        (0.12, 0.40),
        0.76,
        0.17,
        "Common 5-Mb windows\nwithin-arm normalization",
        edgecolor=INK,
        fontsize=7.0,
    )
    workflow_box(
        ax_a,
        (0.12, 0.14),
        0.76,
        0.17,
        r"Spearman $r_s$" "\nwithin-arm shift test",
        edgecolor=INK,
        fontsize=7.0,
    )
    for start, end in (
        ((0.235, 0.69), (0.34, 0.57)),
        ((0.765, 0.69), (0.66, 0.57)),
    ):
        ax_a.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax_a.transAxes,
            textcoords=ax_a.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": INK},
        )
    ax_a.annotate(
        "",
        xy=(0.50, 0.31),
        xytext=(0.50, 0.40),
        xycoords=ax_a.transAxes,
        textcoords=ax_a.transAxes,
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": INK},
    )

    for arm, marker in ARM_MARKERS.items():
        rows = [row for row in supported if row["arm"] == arm]
        ax_b.scatter(
            [row["atlas_normalized"] for row in rows],
            [row["direct_normalized"] for row in rows],
            s=28,
            marker=marker,
            facecolor=BLUE,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.85,
            zorder=3,
            label=arm,
        )
    statistic, pvalue = pedigree["spearman_5mb"]
    limit = max(
        max(float(row["atlas_normalized"]) for row in supported),
        max(float(row["direct_normalized"]) for row in supported),
    ) * 1.08
    ax_b.set_xlim(-0.08, limit)
    ax_b.set_ylim(-0.08, limit)
    ax_b.set_xlabel("Recombination-landscape rate (within-arm relative)")
    ax_b.set_ylabel("Pedigree rate (within-arm relative)")
    ax_b.set_title("Consistent broad-scale spatial ordering", loc="left", pad=5)
    ax_b.text(
        0.04,
        0.96,
        rf"$r_s={statistic:.2f}$" "\n" rf"$P={pvalue:.3f}$" "\n" rf"{pedigree['n_supported_5mb_windows']} windows",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    handles = [
        mpl.lines.Line2D([], [], marker=marker, ls="", markerfacecolor=BLUE, markeredgecolor="white", label=arm)
        for arm, marker in ARM_MARKERS.items()
    ]
    ax_b.legend(
        handles=handles,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.29),
        frameon=False,
        fontsize=7.2,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    null = pedigree_shift_null(pedigree)
    ax_c.hist(null, bins=24, color="#C8C8C8", edgecolor="white", linewidth=0.35)
    ax_c.axvline(statistic, color=BLUE, lw=1.8)
    ax_c.axvline(-statistic, color=BLUE, lw=1.0, ls=(0, (3, 2)))
    ax_c.set_xlim(-0.8, 0.8)
    ax_c.text(
        0.96,
        0.96,
        f"observed $r_s={statistic:.2f}$\n{len(null):,} spatial shifts\ntwo-sided $P={pvalue:.4f}$",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    ax_c.set_xlabel(r"Shifted-map Spearman $r_s$")
    ax_c.set_ylabel("Circular shifts")
    ax_c.set_title("Exact spatial-shift test", loc="left", pad=5)

    for label, axis in zip("abc", (ax_a, ax_b, ax_c), strict=True):
        ps.panel(axis, label, x=-0.14, y=1.10, fontsize=11)
    save(fig, out / "fig_phase2_pedigree")


def format_p(value: float) -> str:
    return "$p < 10^{-3}$" if value < 0.001 else rf"$p={value:.3f}$"


def pyrho_figure(pyrho: dict, arrays_path: Path, out: Path) -> None:
    arrays = np.load(arrays_path)
    rows = pyrho["rows"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 4.6), sharex=True)
    x = (arrays["grid_starts"] + 50_000) / 1e6
    display = {"gamb_BF": "gam·BF", "colu_CI": "col·CI", "gamb_UG": "gam·UG"}
    for ax, row in zip(axes, rows, strict=True):
        cohort = row["cohort"]
        fastrho = arrays[f"{cohort}_fastrho_matched"].astype(float)
        comparator = arrays[f"{cohort}_pyrho"].astype(float)
        fastrho /= np.nanmedian(fastrho)
        comparator /= np.nanmedian(comparator)
        local_x = x[: len(fastrho)]
        ax.plot(local_x, comparator, color=ps.C["pyrho"], lw=1.35, ls=(0, (4, 2.5)), alpha=0.95, label="pyrho")
        ax.plot(local_x, fastrho, color=BLUE, lw=1.5, alpha=0.95, label="fastrho")
        ax.set_yscale("log")
        ax.set_ylim(0.18, 6.0)
        ax.margins(x=0.01)
        ax.set_yticks([0.3, 1, 3])
        ax.set_yticklabels(["0.3", "1", "3"], fontsize=8)
        ax.yaxis.set_minor_formatter(NullFormatter())
        species = "Anopheles coluzzii" if cohort.startswith("colu") else "Anopheles gambiae"
        correlation, pvalue, _n = row["spearman_matched"]
        ax.text(
            0.012,
            0.90,
            f"{display[cohort]}   Spearman $\\rho_s={correlation:.2f}$  ({format_p(pvalue)})",
            transform=ax.transAxes,
            fontsize=8.4,
            va="top",
            color=SPECIES_COLORS[species],
            fontweight="medium",
        )
    axes[len(rows) // 2].set_ylabel("recombination rate / cohort median")
    axes[0].legend(fontsize=8.2, loc="upper right", ncol=2, frameon=False, handlelength=1.3, columnspacing=1.1, handletextpad=0.5)
    axes[-1].set_xlabel(f"position on {rows[0]['arm']} (Mb)")
    fig.suptitle(
        "pyrho recovers the same fine-scale landscape as fastrho (Phase 2, 3R, matched 20-haplotype subsamples)",
        fontsize=10,
        x=0.02,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out / "fig_phase2_pyrho")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    style()
    selection = read_selection(args.selection)
    inversion = json.loads((args.results / "phase2_2la.json").read_text())
    resistance = json.loads((args.results / "phase2_resistance.json").read_text())
    pedigree = json.loads((args.results / "pedigree/phase2_pedigree.json").read_text())
    pyrho = json.loads((args.results / "phase2_pyrho.json").read_text())
    atlas_figure(args.maps, selection, inversion, resistance, args.out)
    pedigree_figure(pedigree, args.out)
    pyrho_figure(pyrho, args.results / "phase2_pyrho.npz", args.out)


if __name__ == "__main__":
    main()
