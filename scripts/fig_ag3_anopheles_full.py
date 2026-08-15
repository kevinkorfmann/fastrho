#!/usr/bin/env python3
"""Build the full 13-cohort Ag3 atlas and resistance-region figure.

This is the Phase 3 extension of the current Phase 2 Figure 4 layout.  It keeps
the manuscript palette and restores the 2La and 15-region resistance panels,
while treating the three-species comparison as descriptive rather than as a
species-permutation claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

import paper_style as ps


ARMS = ("2R", "2L", "3R", "3L", "X")
BREAKPOINTS = (20.524, 42.166)
BLUE = ps.C["fastrho"]
INK = ps.C["truth"]
GRAY = ps.C["relernn"]
NEAR_WHITE = "#FAFAF8"
LIGHT_GRAY = "#E7E7E4"
SPECIES = (
    "Anopheles gambiae",
    "Anopheles coluzzii",
    "Anopheles arabiensis",
)
SPECIES_COLORS = {
    "Anopheles gambiae": BLUE,
    "Anopheles coluzzii": INK,
    "Anopheles arabiensis": GRAY,
}
SPECIES_MARKERS = {
    "Anopheles gambiae": "o",
    "Anopheles coluzzii": "s",
    "Anopheles arabiensis": "^",
}
SPECIES_LABELS = {
    "Anopheles gambiae": r"$A.$ gambiae",
    "Anopheles coluzzii": r"$A.$ coluzzii",
    "Anopheles arabiensis": r"$A.$ arabiensis",
}
SHORT_TO_FULL = {
    "gambiae": "Anopheles gambiae",
    "coluzzii": "Anopheles coluzzii",
    "arabiensis": "Anopheles arabiensis",
}


def style() -> None:
    ps.style()
    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.7,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
        }
    )


def panel(ax: mpl.axes.Axes, label: str, *, x: float = -0.11, y: float = 1.11) -> None:
    ps.panel(ax, label.lower(), x=x, y=y, fontsize=9)


def save(fig: mpl.figure.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_bed(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rows: dict[str, list[tuple[float, float]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            arm, start, end, rate, _cm_mb, _rho = line.rstrip().split("\t")
            midpoint_mb = (float(start) + float(end)) / 2e6
            rows.setdefault(arm, []).append((midpoint_mb, float(rate)))
    return {
        arm: (
            np.asarray([value[0] for value in sorted(values)], dtype=float),
            np.asarray([value[1] for value in sorted(values)], dtype=float),
        )
        for arm, values in rows.items()
    }


def smooth_rows(matrix: np.ndarray, sigma_bins: float = 2.0) -> np.ndarray:
    radius = max(1, int(np.ceil(3 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    smoothed = np.empty_like(matrix, dtype=float)
    for index, row in enumerate(np.asarray(matrix, dtype=float)):
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


def locus_label(locus: str) -> str:
    return locus.replace("Cyp6aa/Cyp6p", "Cyp6aa/p").replace("D7r2/D7r4", "D7r2/r4")


def build_figure(
    atlas_root: Path,
    inversion_path: Path,
    resistance_path: Path,
    output: Path,
) -> None:
    manifest = read_manifest(atlas_root / "manifest.tsv")
    manifest_by_cohort = {row["cohort"]: row for row in manifest}
    raw_maps = {
        cohort: read_bed(atlas_root / "bed" / f"{cohort}.bed")
        for cohort in manifest_by_cohort
    }
    inversion = json.loads(inversion_path.read_text(encoding="utf-8"))
    resistance = json.loads(resistance_path.read_text(encoding="utf-8"))

    map_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    arm_lengths: dict[str, float] = {}
    for cohort, cohort_maps in raw_maps.items():
        map_data[cohort] = {}
        for arm in ARMS:
            positions, rates = cohort_maps[arm]
            positive = rates[(rates > 0) & np.isfinite(rates)]
            normalized = rates / np.nanmedian(positive)
            map_data[cohort][arm] = (positions, normalized)
            arm_lengths[arm] = max(arm_lengths.get(arm, 0.0), float(positions[-1] + 0.025))

    offsets: dict[str, float] = {}
    cursor = 0.0
    for arm in ARMS:
        offsets[arm] = cursor
        cursor += arm_lengths[arm]
    total = cursor

    style()
    fig = plt.figure(figsize=(7.15, 8.30))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=(1.82, 1.22, 1.48, 1.04),
        hspace=0.62,
        left=0.075,
        right=0.985,
        bottom=0.050,
        top=0.985,
    )

    # a | Five-arm atlas for all 13 cohorts, grouped into three species.
    atlas_grid = outer[0].subgridspec(4, 1, height_ratios=(0.66, 1, 1, 1), hspace=0.06)
    label_axis = fig.add_subplot(atlas_grid[0])
    rate_axes = [fig.add_subplot(atlas_grid[index]) for index in (1, 2, 3)]
    label_axis.set_xlim(0, total)
    label_axis.set_ylim(0, 1)
    label_axis.set_axis_off()
    label_axis.set_title(
        "Population-resolved recombination atlas across five chromosome arms",
        loc="left",
        y=1.08,
        pad=3,
    )
    panel(label_axis, "a")

    species_medians: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for axis_index, (species, ax) in enumerate(zip(SPECIES, rate_axes, strict=True)):
        cohorts = [row["cohort"] for row in manifest if row["species"] == species]
        color = SPECIES_COLORS[species]
        for cohort in cohorts:
            for arm in ARMS:
                position, rates = map_data[cohort][arm]
                ax.plot(offsets[arm] + position, rates, color=color, alpha=0.18, lw=0.30)
        for arm in ARMS:
            arrays = [map_data[cohort][arm][1] for cohort in cohorts]
            length = min(map(len, arrays))
            position = map_data[cohorts[0]][arm][0][:length]
            median = np.nanmedian(np.asarray([array[:length] for array in arrays]), axis=0)
            species_medians[(species, arm)] = (position, median)
            ax.plot(offsets[arm] + position, median, color=color, lw=0.95)
        ax.set_ylabel(SPECIES_LABELS[species], color=color, fontsize=5.9, labelpad=4)
        ax.set_xlim(0, total)
        ax.set_ylim(0, 12)
        ax.grid(axis="y", color="0.92", lw=0.45)
        ax.tick_params(axis="x", labelbottom=axis_index == len(rate_axes) - 1)

    for ax in rate_axes:
        for boundary in (offsets["2L"] + BREAKPOINTS[0], offsets["2L"] + BREAKPOINTS[1]):
            ax.axvline(boundary, color="0.58", lw=0.55, ls=(0, (2.2, 2.2)), zorder=-1)
        for arm in ARMS[1:]:
            ax.axvline(offsets[arm], color="0.84", lw=0.5, zorder=-1)

    label_transform = mtransforms.blended_transform_factory(label_axis.transData, label_axis.transAxes)
    lane_right = [-np.inf, -np.inf, -np.inf]
    lane_y = (0.07, 0.40, 0.73)
    ordered_loci = sorted(
        resistance["loci"].items(),
        key=lambda item: offsets[item[1]["arm"]] + float(item[1]["mb"]),
    )
    for locus, target in ordered_loci:
        arm = target["arm"]
        mb = float(target["mb"])
        genome_x = offsets[arm] + mb
        for species, ax in zip(SPECIES, rate_axes, strict=True):
            position, median = species_medians[(species, arm)]
            local = np.abs(position - mb) <= 0.15
            local_rate = (
                float(np.nanmedian(median[local]))
                if np.any(local)
                else float(median[int(np.nanargmin(np.abs(position - mb)))])
            )
            ax.axvline(genome_x, color=INK, alpha=0.18, lw=0.45, zorder=-1)
            ax.scatter(
                genome_x,
                min(local_rate, 11.65),
                marker="^",
                s=21,
                facecolor=INK,
                edgecolor="white",
                linewidth=0.50,
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
            [0, lane_y[lane] - 0.04],
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
            fontsize=4.6,
            fontweight="bold",
            clip_on=False,
        )

    centers = [offsets[arm] + arm_lengths[arm] / 2 for arm in ARMS]
    rate_axes[-1].set_xticks(centers, ARMS)
    rate_axes[-1].set_xlabel("Chromosome arm")
    rate_axes[0].text(offsets["2L"] + np.mean(BREAKPOINTS), 11.2, "2La", ha="center", color=INK)

    # b-c | 2La landmark analysis across all 13 cohorts.
    validation_grid = outer[1].subgridspec(1, 2, width_ratios=(1.36, 1.0), wspace=0.48)
    validation_rows = []
    for cohort, record in inversion.items():
        if cohort.startswith("_"):
            continue
        validation_rows.append(
            {
                "cohort": cohort,
                "species": manifest_by_cohort[cohort]["species"],
                "H": float(record["2La"]["H"]),
            }
        )
    validation_rows.sort(key=lambda row: row["H"])

    ax_heat = fig.add_subplot(validation_grid[0, 0])
    heat_sources = []
    depths = []
    for row in validation_rows:
        positions, rates = raw_maps[row["cohort"]]["2L"]
        outside = (positions < BREAKPOINTS[0]) | (positions > BREAKPOINTS[1])
        baseline = float(np.nanmedian(rates[outside & np.isfinite(rates) & (rates > 0)]))
        ratio = rates / baseline
        values = np.log2(np.maximum(ratio, 2**-3))
        inside = (positions >= BREAKPOINTS[0]) & (positions <= BREAKPOINTS[1])
        depth = 1.0 - float(np.nanmedian(rates[inside]) / baseline)
        depths.append(depth)
        heat_sources.append((positions, values))
    shared_start = max(source[0][0] for source in heat_sources)
    shared_end = min(source[0][-1] for source in heat_sources)
    positions = np.arange(
        np.ceil(shared_start * 20) / 20,
        np.floor(shared_end * 20) / 20 + 0.025,
        0.05,
    )
    heat = [np.interp(positions, row_positions, values) for row_positions, values in heat_sources]
    heat_image = ax_heat.imshow(
        smooth_rows(np.asarray(heat)),
        aspect="auto",
        interpolation="nearest",
        extent=(positions[0], positions[-1], len(validation_rows) - 0.5, -0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2, vcenter=0, vmax=2),
    )
    for boundary in BREAKPOINTS:
        ax_heat.axvline(boundary, color=INK, lw=0.65)
    ax_heat.set_yticks(
        np.arange(len(validation_rows)),
        [f"{row['cohort']}  H={row['H']:.2f}" for row in validation_rows],
    )
    for tick, row in zip(ax_heat.get_yticklabels(), validation_rows, strict=True):
        tick.set_color(SPECIES_COLORS[row["species"]])
    ax_heat.set_xlabel("Chromosome 2L position (Mb)")
    ax_heat.set_title("2La cold-block depth varies among cohorts", loc="left")
    heat_colorbar = fig.colorbar(heat_image, ax=ax_heat, fraction=0.035, pad=0.02)
    heat_colorbar.set_label(r"$\log_2$ rate ratio")
    panel(ax_heat, "b")

    ax_prediction = fig.add_subplot(validation_grid[0, 1])
    for row, depth in zip(validation_rows, depths, strict=True):
        species = row["species"]
        ax_prediction.scatter(
            row["H"],
            depth,
            s=28,
            marker=SPECIES_MARKERS[species],
            color=SPECIES_COLORS[species],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    h_values = np.asarray([row["H"] for row in validation_rows])
    depth_values = np.asarray(depths)
    slope, intercept = np.polyfit(h_values, depth_values, 1)
    xx = np.linspace(0, 0.51, 100)
    ax_prediction.plot(xx, slope * xx + intercept, color="0.25", lw=1.0)
    ax_prediction.axhline(0, color="0.6", ls="--", lw=0.7)
    ax_prediction.set_xlim(-0.02, 0.52)
    ax_prediction.set_xlabel("Expected 2La heterokaryotype frequency")
    ax_prediction.set_ylabel(r"Suppression depth $1-r_{\rm in}/r_{\rm out}$")
    pearson = pearsonr(h_values, depth_values)
    spearman = spearmanr(h_values, depth_values)
    ax_prediction.text(
        0.03,
        0.97,
        "Pearson $r=%.2f$, $P=%.3f$\nSpearman $r_s=%.2f$, $P=%.3f$\n$n=%d$ cohorts"
        % (pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue, len(validation_rows)),
        transform=ax_prediction.transAxes,
        va="top",
        fontsize=6.2,
    )
    legend_handles = [
        Line2D(
            [],
            [],
            marker=SPECIES_MARKERS[species],
            ls="",
            markerfacecolor=SPECIES_COLORS[species],
            markeredgecolor="white",
            label=SPECIES_LABELS[species].replace("$", ""),
        )
        for species in SPECIES
    ]
    ax_prediction.legend(handles=legend_handles, loc="lower right", frameon=False, handletextpad=0.2)
    ax_prediction.set_title("Suppression follows arrangement mixing", loc="left")
    panel(ax_prediction, "c")

    # d | Complete 15-region by 13-cohort resistance heatmap.
    order_index = {species: index for index, species in enumerate(SPECIES)}
    ordered_rows = sorted(
        resistance["rows"],
        key=lambda row: (order_index[SHORT_TO_FULL[row["species"]]], row["cohort"]),
    )
    loci = list(resistance["loci"])
    resistance_grid = outer[2].subgridspec(1, 3, width_ratios=(5.2, 0.18, 1.0), wspace=0.20)
    ax_resistance = fig.add_subplot(resistance_grid[0, 0])
    log2_ratios = np.asarray(
        [
            [np.log2(row["loci"][locus]["ratio_vs_haplotype_matched"]) for row in ordered_rows]
            for locus in loci
        ]
    )
    cold_warm = LinearSegmentedColormap.from_list(
        "ag3_cold_equal_warm",
        [BLUE, NEAR_WHITE, GRAY],
    )
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
        tick.set_color(SPECIES_COLORS[SHORT_TO_FULL[row["species"]]])
    ax_resistance.set_yticks(np.arange(len(loci)), [locus_label(locus) for locus in loci])
    ax_resistance.axvline(5.5, color="0.15", lw=1.0)
    ax_resistance.axvline(9.5, color="0.15", lw=1.0)
    ax_resistance.set_title("Resistance-region rates relative to haplotype-matched controls", loc="left")
    panel(ax_resistance, "d")

    color_axis = fig.add_subplot(resistance_grid[0, 1])
    resistance_colorbar = fig.colorbar(resistance_image, cax=color_axis)
    resistance_colorbar.set_ticks([-4, -2, 0, 2])
    resistance_colorbar.set_ticklabels(["≤−4", "−2", "0", "2"])
    color_axis.set_title(r"$\log_2$" "\nratio", fontsize=6.0, pad=3)

    count_axis = fig.add_subplot(resistance_grid[0, 2], sharey=ax_resistance)
    cold_counts = np.sum(log2_ratios < 0, axis=1)
    y = np.arange(len(loci))
    count_axis.barh(y, cold_counts, color=BLUE, height=0.64)
    for yi, count in zip(y, cold_counts, strict=True):
        inside = count >= 8
        count_axis.text(
            count - 0.18 if inside else count + 0.18,
            yi,
            f"{count}/13",
            ha="right" if inside else "left",
            va="center",
            fontsize=5.4,
            color="white" if inside else INK,
        )
    count_axis.set_xlim(0, 13)
    count_axis.set_xticks([0, 4, 8, 13])
    count_axis.tick_params(axis="y", left=False, labelleft=False)
    count_axis.set_xlabel("Cohorts with\nratio < 1")
    count_axis.set_title("Direction in cohorts", loc="left", fontsize=6.5)
    count_axis.grid(axis="x", color="0.90", lw=0.5)

    # e-f | Cohort-level and descriptive species-level summaries.
    controls_grid = outer[3].subgridspec(1, 2, width_ratios=(0.96, 1.44), wspace=0.58)
    ax_cohorts = fig.add_subplot(controls_grid[0, 0])
    cohort_rows = sorted(resistance["rows"], key=lambda row: row["ratio_vs_haplotype_matched"])
    y = np.arange(len(cohort_rows))
    for yi, row in zip(y, cohort_rows, strict=True):
        species = SHORT_TO_FULL[row["species"]]
        significant = row["perm_p"] < 0.05
        color = SPECIES_COLORS[species]
        ax_cohorts.plot(
            np.log2(row["ratio_vs_haplotype_matched"]),
            yi,
            marker=SPECIES_MARKERS[species],
            ms=4.8,
            markerfacecolor=color if significant else "white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            ls="",
        )
    ax_cohorts.axvline(0, color="0.45", lw=0.8)
    ax_cohorts.set_yticks(y, [row["cohort"] for row in cohort_rows])
    for tick, row in zip(ax_cohorts.get_yticklabels(), cohort_rows, strict=True):
        tick.set_color(SPECIES_COLORS[SHORT_TO_FULL[row["species"]]])
    ax_cohorts.set_xlabel(r"Cohort median $\log_2$(focal/control)")
    ax_cohorts.set_title("Population-level resistance summary", loc="left")
    ax_cohorts.text(
        0.02,
        0.02,
        "filled: permutation $P<0.05$",
        transform=ax_cohorts.transAxes,
        fontsize=5.7,
        va="bottom",
    )
    panel(ax_cohorts, "e", x=-0.16)

    ax_species = fig.add_subplot(controls_grid[0, 1])
    y_positions = np.arange(len(SPECIES) - 1, -1, -1, dtype=float)
    for index, species in enumerate(SPECIES):
        short = species.split()[-1]
        group = [
            float(row["ratio_vs_haplotype_matched"])
            for row in resistance["rows"]
            if row["species"] == short
        ]
        median = float(np.median(group))
        interval = bootstrap_median_ci(group, 1000 + index)
        jitter = np.linspace(-0.12, 0.12, len(group)) if len(group) > 1 else np.zeros(1)
        ax_species.scatter(
            group,
            y_positions[index] + jitter,
            s=23,
            marker=SPECIES_MARKERS[species],
            facecolor="white",
            edgecolor=SPECIES_COLORS[species],
            linewidth=0.9,
            zorder=3,
        )
        ax_species.errorbar(
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
        ax_species.text(
            median,
            y_positions[index] + 0.27,
            f"median {median:.2f}",
            color=SPECIES_COLORS[species],
            fontsize=6.0,
            fontweight="bold",
            ha="center",
            va="bottom",
        )
    ax_species.axvline(1, color=INK, lw=0.9)
    ax_species.set_yticks(
        y_positions,
        [
            rf"$\it{{A.\ {species.split()[-1]}}}$  ($n={sum(row['species'] == species.split()[-1] for row in resistance['rows'])}$)"
            for species in SPECIES
        ],
    )
    ax_species.set_xlim(0.35, 1.10)
    ax_species.set_ylim(-0.45, 2.55)
    ax_species.set_xlabel("Resistance-region rate / matched-control rate")
    ax_species.set_title("Fifteen-region summary by species (descriptive)", loc="left")
    ax_species.spines["left"].set_visible(False)
    ax_species.tick_params(axis="y", length=0)
    panel(ax_species, "f", x=-0.38)

    save(fig, output)
    print(f"wrote {output.with_suffix('.pdf')}")
    print(f"wrote {output.with_suffix('.png')}")
    print(
        "2La: Pearson r=%.4f P=%.4g; Spearman r=%.4f P=%.4g"
        % (pearson.statistic, pearson.pvalue, spearman.statistic, spearman.pvalue)
    )
    print(
        "resistance: median=%.4f; significant cohorts=%d/%d"
        % (
            resistance["median_ratio_all"],
            resistance["n_sig_total"],
            len(resistance["rows"]),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path("legacy/pre-phase2-snapshot/atlas/anopheles"),
    )
    parser.add_argument(
        "--inversion",
        type=Path,
        default=Path("legacy/pre-phase2-snapshot/paper/figdata/agam_inv_freq.json"),
    )
    parser.add_argument(
        "--resistance",
        type=Path,
        default=Path(
            "legacy/pre-phase2-snapshot/paper/figdata/"
            "agam_haplotype_matched_controls_hancock_mechanisms.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/fig_ag3_anopheles_full.pdf"),
    )
    args = parser.parse_args()
    build_figure(args.atlas_root, args.inversion, args.resistance, args.output)


if __name__ == "__main__":
    main()
