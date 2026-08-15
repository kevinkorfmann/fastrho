"""Generate the deterministic multipanel figures for the manuscript draft."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "manuscript" / "figures"
BLUE = "#2737E7"
GREEN = "#151515"
ORANGE = "#777777"
PURPLE = "#151515"
GRAY = "#777777"
SPECIES = {
    "gambiae": BLUE,
    "coluzzii": GREEN,
    "arabiensis": ORANGE,
}
SPECIES_LONG = {
    "Anopheles gambiae": ("gambiae", BLUE),
    "Anopheles coluzzii": ("coluzzii", GREEN),
    "Anopheles arabiensis": ("arabiensis", ORANGE),
}
ARMS = ["2R", "2L", "3R", "3L", "X"]
ARM_LENGTH_MB = {"2R": 61.5, "2L": 49.4, "3R": 53.2, "3L": 42.0, "X": 24.4}
LOCUS_PURPLE = "#151515"
LOCUS_SHORT = {
    "Vgsc/kdr": "Vgsc/kdr",
    "Rdl": "Rdl",
    "Ace1": "Ace1",
    "Gste2": "Gste2",
    "Cyp6aa/Cyp6p": "Cyp6aa/p",
    "Cyp9k1": "Cyp9k1",
    "D7r2/D7r4": "D7r2/r4",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7.4,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#151515",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
        }
    )


def panel_label(axis, label: str) -> None:
    axis.text(-0.11, 1.11, label.lower(), transform=axis.transAxes, weight="bold", size=9)


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def smooth_rows(matrix: np.ndarray, sigma_bins: float = 1.5) -> np.ndarray:
    """Gaussian display smoother that preserves missing values and row identities."""

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


def paired_record(demography: dict, key: str, method: str, history: str) -> dict:
    scenario = key.removesuffix("_n20")
    if method == "fastrho":
        return demography["scenarios"][scenario]["fastrho_reference"]["25kb"]
    return demography["scenarios"][scenario][method]["arms"][history]["25kb"]


def figure1(
    summary: dict, selection: dict, selection_windows: np.lib.npyio.NpzFile, demography: dict
) -> None:
    """Benchmark qualification as a standalone five-panel figure."""

    fig = plt.figure(figsize=(7.15, 4.28))
    outer = fig.add_gridspec(2, 1, hspace=0.78)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.55, 1.00, 1.25], wspace=0.54)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.95, 1.30], wspace=0.46)

    ax = fig.add_subplot(top[0, 0])
    scenarios = [
        ("Constant", "const_n20"),
        ("Bottleneck", "bottleneck_n20"),
        ("Expansion", "expansion_n20"),
        ("deCODE", "real_decode"),
        ("HapMap", "real_hapmap"),
        ("Dog", "real_dog"),
    ]
    methods = [("fastrho", BLUE, "o"), ("pyrho", GREEN, "s"), ("ReLERNN", GRAY, "^")]
    x = np.arange(len(scenarios))
    for method, color, marker in methods:
        values = []
        for _label, key in scenarios:
            method_key = method.lower()
            if key in {"bottleneck_n20", "expansion_n20"}:
                record = paired_record(demography, key, method_key, "matched")
            else:
                record = summary[key]["scales"]["25kb"].get(method_key)
            if record is None or not np.isfinite(record.get("pearson", np.nan)):
                raise ValueError(
                    "Figure 1B requires a matched, finite benchmark value for "
                    f"{method} in {key}; missing benchmark cells may not be "
                    "plotted as unavailable."
                )
            values.append(record["pearson"])
        ax.plot(
            x,
            values,
            color=color,
            marker=marker,
            ms=4.5,
            lw=1.1,
            label=method,
            zorder={"fastrho": 4, "pyrho": 3, "ReLERNN": 2}[method],
        )
        label_offset = {"fastrho": 0.028, "pyrho": -0.028}.get(method, 0.0)
        ax.text(
            x[-1] + 0.16,
            values[-1] + label_offset,
            method,
            color=color,
            fontsize=5.8,
            va="center",
            clip_on=False,
        )
        if method_key in {"pyrho", "relernn"}:
            for scenario_index, (_label, key) in enumerate(scenarios[1:3], start=1):
                constant = paired_record(demography, key, method_key, "constant")["pearson"]
                matched = paired_record(demography, key, method_key, "matched")["pearson"]
                constant_x = scenario_index - 0.09
                ax.plot(
                    [constant_x, scenario_index],
                    [constant, matched],
                    color=color,
                    lw=0.7,
                    ls=":",
                    alpha=0.75,
                    zorder=1,
                )
                ax.scatter(
                    constant_x,
                    constant,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    s=20,
                    linewidth=0.9,
                    zorder=4,
                )
    ax.set_xticks(x, [label for label, _key in scenarios], rotation=38, ha="right")
    ax.set_xlim(-0.28, len(scenarios) + 0.32)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Pearson $r$ at 25 kb")
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                markersize=4.4,
                label="constant",
            ),
            Line2D(
                [],
                [],
                color="black",
                marker="o",
                markerfacecolor="black",
                linestyle="none",
                markersize=4.4,
                label="matched",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.95,
        fontsize=5.1,
        handletextpad=0.35,
        columnspacing=0.70,
        borderaxespad=0,
    )
    ax.set_title("Reconstruction of map shape at 25 kb", loc="left")
    panel_label(ax, "a")

    ax = fig.add_subplot(top[0, 1])
    nominal = np.asarray(summary["heldout"]["coverage_curve"]["nominal"])
    empirical = np.asarray(summary["heldout"]["coverage_curve"]["empirical"])
    ax.plot([0.45, 1.0], [0.45, 1.0], color="0.6", ls="--", lw=1, label="ideal")
    ax.plot(nominal, empirical, color=BLUE, marker="o", ms=4, lw=1.4)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration of intervals", loc="left")
    panel_label(ax, "b")

    ax = fig.add_subplot(top[0, 2])
    x = np.arange(len(scenarios))
    offsets = (-0.22, 0.0, 0.22)
    for offset, (method, color, marker) in zip(offsets, methods):
        values = []
        for _label, key in scenarios:
            method_key = method.lower()
            if key in {"bottleneck_n20", "expansion_n20"}:
                record = paired_record(demography, key, method_key, "matched")
            else:
                record = summary[key]["scales"]["25kb"].get(method_key)
            values.append(np.nan if record is None else record["bias_ratio"])
        ax.scatter(x + offset, values, color=color, marker=marker, s=18, zorder=3, label=method)
        if method_key in {"pyrho", "relernn"}:
            for scenario_index, (_label, key) in enumerate(scenarios[1:3], start=1):
                constant = paired_record(demography, key, method_key, "constant")["bias_ratio"]
                matched = paired_record(demography, key, method_key, "matched")["bias_ratio"]
                position = scenario_index + offset
                constant_x = position - 0.07
                ax.plot(
                    [constant_x, position],
                    [constant, matched],
                    color=color,
                    lw=0.7,
                    ls=":",
                    alpha=0.75,
                )
                ax.scatter(
                    constant_x,
                    constant,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    s=18,
                    linewidth=0.9,
                    zorder=4,
                )
    ax.axhline(1, color="0.25", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 7.2)
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x, [label for label, _key in scenarios], rotation=42, ha="right")
    ax.set_ylabel("median estimated / true rate")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.set_title("Calibration of rates", loc="left")
    panel_label(ax, "c")

    ax = fig.add_subplot(bottom[0, 0])
    timing = summary["timings"]
    method_styles = [("fastrho", BLUE, "o"), ("pyrho", GREEN, "s"), ("relernn", GRAY, "^")]
    for name, color, marker in method_styles:
        score = summary["const_n20"]["scales"]["25kb"].get(name)
        if score is None:
            continue
        ax.scatter(timing[name], score["pearson"], color=color, marker=marker, s=40, zorder=3)
        label = "ReLERNN" if name == "relernn" else name
        ax.annotate(label, (timing[name], score["pearson"]), xytext=(4, 3), textcoords="offset points", color=color)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 2.0e4)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("relative wall-clock cost per dataset")
    ax.set_ylabel("Pearson $r$ at 25 kb")
    ax.grid(color="0.91", lw=0.5)
    ax.text(0.04, 0.06, r"faster  $\leftarrow$", transform=ax.transAxes, color="0.4")
    ax.set_title("Cost-accuracy trade-off", loc="left")
    panel_label(ax, "d")

    linked = bottom[0, 1].subgridspec(1, 2, wspace=0.42)
    ax_shape = fig.add_subplot(linked[0, 0])
    records = {record["name"]: record for record in selection["conditions"]}
    shape_conditions = [
        ("neutral", records["neutral"]),
        ("BGS", records["bgsint_4"]),
        ("hard\nsweep", records["compl_7"]),
    ]
    xx = np.arange(len(shape_conditions))
    for method, color, marker in (("fastrho", BLUE, "o"), ("pyrho", GREEN, "s")):
        key = "fastrho_cmn_25kb" if method == "fastrho" else "pyrho_25kb"
        values = [record[key][0] for _label, record in shape_conditions]
        ax_shape.plot(xx, values, color=color, marker=marker, lw=1.1, ms=4, label=method)
    ax_shape.set_xticks(xx, [label for label, _record in shape_conditions])
    ax_shape.set_ylim(0.68, 0.93)
    ax_shape.set_ylabel("Pearson $r$ at 25 kb")
    ax_shape.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax_shape.set_title("SLiM: map shape", loc="left")
    panel_label(ax_shape, "e")

    ax_scale = fig.add_subplot(linked[0, 1])
    scale_conditions = ("neutral", "sweep")
    scale_labels = ("neutral", "hard\nsweep")
    xx = np.arange(len(scale_conditions))
    for method, color, marker in (("fastrho", BLUE, "o"), ("pyrho", GREEN, "s")):
        values = []
        for condition in scale_conditions:
            truth = np.asarray(selection_windows[f"calib_true_{condition}"], dtype=float)
            predicted = np.asarray(
                selection_windows[f"calib_{method}_{condition}"], dtype=float
            )
            keep = (
                np.isfinite(truth)
                & np.isfinite(predicted)
                & (truth > 0)
                & (predicted > 0)
            )
            values.append(float(np.median(predicted[keep] / truth[keep])))
        ax_scale.plot(xx, values, color=color, marker=marker, lw=1.1, ms=4)
    ax_scale.axhline(1, color="0.4", lw=0.8, ls=":")
    ax_scale.set_xticks(xx, scale_labels)
    ax_scale.set_ylim(0.45, 1.03)
    ax_scale.set_ylabel("median estimated / true")
    ax_scale.text(
        0.97,
        0.87,
        "1 = correct scale",
        transform=ax_scale.transAxes,
        ha="right",
        va="top",
        fontsize=5.3,
        color="0.35",
    )
    ax_scale.set_title("SLiM: rate scale", loc="left")
    save(fig, "fig1_method_validation")


def load_manifest() -> dict:
    with (ROOT / "atlas" / "anopheles" / "manifest.tsv").open() as handle:
        return {row["cohort"]: row for row in csv.DictReader(handle, delimiter="\t")}


def load_bed(cohort: str) -> dict:
    rows = {}
    path = ROOT / "atlas" / "anopheles" / "bed" / f"{cohort}.bed"
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            arm, start, _end, rate, _cm, _rho = line.rstrip().split("\t")
            rows.setdefault(arm, []).append((float(start) / 1e6, float(rate) * 1e8))
    return {arm: np.asarray(values, float) for arm, values in rows.items()}


def arm_offsets() -> dict:
    values = {}
    offset = 0.0
    for arm in ARMS:
        values[arm] = offset
        offset += ARM_LENGTH_MB[arm]
    return values


def figure2(validation: dict, resistance_panel: dict) -> None:
    """Genome-wide atlas plus the independent 2La positive-control structure."""

    manifest = load_manifest()
    data = {cohort: load_bed(cohort) for cohort in manifest}
    offsets = arm_offsets()
    fig = plt.figure(figsize=(7.15, 6.55))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.12, 0.88], hspace=0.44)
    top = outer[0].subgridspec(4, 1, height_ratios=[0.92, 1, 1, 1], hspace=0.08)
    label_axis = fig.add_subplot(top[0])
    axes = [fig.add_subplot(top[index]) for index in range(1, 4)]
    total = sum(ARM_LENGTH_MB.values())
    label_axis.set_xlim(0, total)
    label_axis.set_ylim(0, 1)
    label_axis.set_axis_off()
    label_axis.set_title("50-kb maps for 13 cohorts across five chromosome arms", loc="left", y=1.08, pad=5)
    panel_label(label_axis, "a")
    species_medians = {}
    for index, (species_long, (species_short, color)) in enumerate(SPECIES_LONG.items()):
        ax = axes[index]
        cohorts = [c for c, row in manifest.items() if row["species"] == species_long]
        for cohort in cohorts:
            for arm in ARMS:
                values = data[cohort][arm]
                ax.plot(offsets[arm] + values[:, 0], values[:, 1], color=color, alpha=0.18, lw=0.35)
        for arm in ARMS:
            arrays = [data[cohort][arm][:, 1] for cohort in cohorts]
            length = min(map(len, arrays))
            position = data[cohorts[0]][arm][:length, 0]
            median = np.nanmedian(np.asarray([array[:length] for array in arrays]), axis=0)
            species_medians[(species_short, arm)] = (position, median)
            ax.plot(offsets[arm] + position, median, color=color, lw=0.9)
        ax.set_ylabel(f"$A.$ {species_short}", color=color, fontsize=6.0, labelpad=5)
        ax.set_xlim(0, total)
        ax.set_ylim(0, 12)
        ax.grid(axis="y", color="0.92", lw=0.5)
    for ax in axes:
        for boundary in (
            offsets["2L"] + 20.524,
            offsets["2L"] + 42.166,
            offsets["2R"] + 19.024,
            offsets["2R"] + 26.759,
        ):
            ax.axvline(
                boundary,
                color="0.58",
                lw=0.55,
                ls=(0, (2.2, 2.2)),
                zorder=-1,
            )
        for arm in ARMS[1:]:
            ax.axvline(offsets[arm], color="0.84", lw=0.55, zorder=-1)

    # Narrow locus bands identify the exact windows used in the resistance
    # analysis. Each triangle sits at that species' median local rate, retaining
    # the biological position and the quantitative trough height together.
    loci = {
        record["locus"]: record
        for record in resistance_panel["definitions"]
    }
    label_transform = mtransforms.blended_transform_factory(label_axis.transData, label_axis.transAxes)
    ordered_loci = sorted(
        loci.items(),
        key=lambda item: offsets[item[1]["arm"]] + float(item[1]["mb"]),
    )
    label_lane_right = [-np.inf, -np.inf, -np.inf]
    label_y = [0.12, 0.46, 0.80]
    for locus, record in ordered_loci:
        arm = record["arm"]
        mb = float(record["mb"])
        genome_x = offsets[arm] + mb
        for (species_short, _color), ax in zip(SPECIES_LONG.values(), axes):
            position, median = species_medians[(species_short, arm)]
            local = np.abs(position - mb) <= 0.15
            if np.any(local):
                local_rate = float(np.nanmedian(median[local]))
            else:
                nearest = int(np.nanargmin(np.abs(position - mb)))
                local_rate = float(median[nearest])
            ax.axvline(
                genome_x,
                color=LOCUS_PURPLE,
                alpha=0.40,
                lw=0.45,
                zorder=-1,
            )
            ax.scatter(
                genome_x,
                min(local_rate, 11.65),
                marker="^",
                s=31,
                facecolor=LOCUS_PURPLE,
                edgecolor="white",
                linewidth=0.65,
                zorder=6,
                clip_on=False,
            )
        short_label = LOCUS_SHORT.get(locus, locus)
        half_width = max(2.7, 0.62 * len(short_label))
        label_x = float(np.clip(genome_x, half_width, total - half_width))
        lane = next(
            (
                index
                for index, previous_right in enumerate(label_lane_right)
                if label_x - half_width >= previous_right + 1.2
            ),
            int(np.argmin(label_lane_right)),
        )
        label_lane_right[lane] = label_x + half_width
        label_axis.plot(
            [genome_x, label_x],
            [0, label_y[lane] - 0.06],
            color=LOCUS_PURPLE,
            alpha=0.30,
            lw=0.4,
            clip_on=False,
        )
        label_axis.text(
            label_x,
            label_y[lane],
            short_label,
            transform=label_transform,
            ha="center",
            va="bottom",
            color=LOCUS_PURPLE,
            fontsize=5.0,
            fontweight="bold",
            clip_on=False,
        )
    centers = [offsets[arm] + ARM_LENGTH_MB[arm] / 2 for arm in ARMS]
    axes[-1].set_xticks(centers, ARMS)
    axes[-1].set_xlabel("Chromosome arm")
    axes[0].text(offsets["2L"] + 31.3, 11.3, "2La", ha="center", color="0.35")
    axes[0].text(offsets["2R"] + 22.9, 11.3, "2Rb", ha="center", color="0.4")
    axes[-1].legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="^",
                ls="",
                mfc=LOCUS_PURPLE,
                mec="white",
                mew=0.65,
                ms=6,
                label="resistance region (local median)",
            )
        ],
        frameon=False,
        fontsize=5.8,
        loc="upper left",
        borderaxespad=0.15,
    )
    middle = outer[1].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.52)
    rows = sorted(validation["rows"], key=lambda row: row["het_expected"])
    ax = fig.add_subplot(middle[0, 0])
    heat = []
    positions = None
    for row in rows:
        values = data[row["pop"]]["2L"]
        positions = values[:, 0]
        rates = values[:, 1]
        outside = (positions < 20.524) | (positions > 42.166)
        heat.append(np.log2(np.maximum(rates / np.nanmedian(rates[outside]), 2 ** -3)))
    matrix = np.asarray(heat)
    display_matrix = smooth_rows(matrix, sigma_bins=2.0)
    image = ax.imshow(
        display_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=(positions[0], positions[-1], len(rows) - 0.5, -0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
    )
    ax.axvline(20.524, color="black", lw=0.65)
    ax.axvline(42.166, color="black", lw=0.65)
    ax.set_yticks(np.arange(len(rows)), [f"{row['pop']}  H={row['het_expected']:.2f}" for row in rows])
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(SPECIES[row["taxon"]])
    label_transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        (20.524 + 42.166) / 2,
        1.015,
        "2La",
        transform=label_transform,
        ha="center",
        va="bottom",
        fontsize=6,
        fontweight="bold",
    )
    ax.set_xlabel("chromosome 2L position (Mb)")
    ax.set_title("cohort rate relative to its collinear-arm median", loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label(r"$\log_2$ rate ratio")
    panel_label(ax, "b")

    ax = fig.add_subplot(middle[0, 1])
    for row in validation["rows"]:
        ax.scatter(
            row["het_expected"], row["suppression_depth"], s=24,
            color=SPECIES[row["taxon"]], edgecolor="white", linewidth=0.5,
        )
    h = np.asarray([row["het_expected"] for row in validation["rows"]])
    depth = np.asarray([row["suppression_depth"] for row in validation["rows"]])
    slope, intercept = np.polyfit(h, depth, 1)
    xx = np.linspace(0, 0.51, 100)
    ax.plot(xx, slope * xx + intercept, color="0.25", lw=1.0)
    ax.axhline(0, color="0.6", ls="--", lw=0.7)
    ax.set_xlabel("full-cohort expected 2La heterokaryotype frequency")
    ax.set_ylabel(r"suppression depth $1-r_{in}/r_{out}$")
    ax.set_xlim(-0.02, 0.52)
    ax.text(
        0.03, 0.97,
        "Pearson $r=%.2f$, $P=%.3f$\nSpearman $r_s=%.2f$, $P=%.3f$\n$n=%d$ cohorts"
        % (
            validation["pearson_Hexp_depth"][0],
            validation["pearson_Hexp_depth"][1],
            validation["spearman_Hexp_depth"][0],
            validation["spearman_Hexp_depth"][1],
            len(validation["rows"]),
        ),
        transform=ax.transAxes, va="top", size=6.4,
    )
    ax.set_title("suppression vs heterokaryotype frequency", loc="left")
    panel_label(ax, "c")

    bottom = outer[2].subgridspec(1, 2, wspace=0.40)
    ax = fig.add_subplot(bottom[0, 0])
    common_lengths = {
        arm: min(len(data[cohort][arm]) for cohort in manifest)
        for arm in ARMS
    }
    matrix = []
    cohorts = list(manifest)
    for cohort in cohorts:
        vector = np.concatenate([
            np.log10(np.clip(data[cohort][arm][:common_lengths[arm], 1], 1e-6, None))
            for arm in ARMS
        ])
        matrix.append(vector)
    matrix = np.asarray(matrix)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    u, singular, _vt = np.linalg.svd(matrix, full_matrices=False)
    pcs = u[:, :2] * singular[:2]
    variance = singular ** 2 / np.sum(singular ** 2)
    for index, cohort in enumerate(cohorts):
        species_short = SPECIES_LONG[manifest[cohort]["species"]][0]
        ax.scatter(pcs[index, 0], pcs[index, 1], color=SPECIES[species_short], s=22, edgecolor="white", lw=0.5)
        ax.annotate(cohort, (pcs[index, 0], pcs[index, 1]), xytext=(3, 2), textcoords="offset points", size=5.2)
    ax.set_xlabel(f"PC1 ({100 * variance[0]:.0f}%)")
    ax.set_ylabel(f"PC2 ({100 * variance[1]:.0f}%)")
    ax.set_title("log-rate profiles retain population structure", loc="left")
    panel_label(ax, "d")

    ax = fig.add_subplot(bottom[0, 1])
    africa = [
        (-17, 15), (-17, 21), (-10, 30), (0, 36), (10, 37), (20, 32),
        (32, 31), (37, 22), (43, 12), (51, 11), (41, -2), (40, -10),
        (35, -18), (28, -33), (20, -35), (15, -28), (12, -18), (9, -1),
        (9, 4), (5, 5), (-4, 5), (-8, 5), (-13, 9), (-16, 13), (-17, 15),
    ]
    ax.fill([point[0] for point in africa], [point[1] for point in africa], color="#F0EFEA", ec="#C8C3B7", lw=0.7)
    for cohort, record in manifest.items():
        species_short = SPECIES_LONG[record["species"]][0]
        ax.scatter(
            float(record["lon"]), float(record["lat"]),
            s=18 + 58 * float(record["twoLa_p"]), color=SPECIES[species_short],
            edgecolor="white", lw=0.5, alpha=0.92,
        )
    ax.set_xlim(-20, 54)
    ax.set_ylim(-38, 40)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(r"geographic sampling; size $\propto$ map-panel 2La frequency", loc="left")
    panel_label(ax, "e")
    save(fig, "fig2_anopheles_atlas")


def figure3(validation: dict) -> None:
    rows = sorted(validation["rows"], key=lambda row: row["het_expected"])
    fig = plt.figure(figsize=(7.15, 3.45))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.62)

    ax = fig.add_subplot(grid[0, 0])
    heat = []
    positions = None
    for row in rows:
        values = load_bed(row["pop"])["2L"]
        positions = values[:, 0]
        rates = values[:, 1]
        outside = (positions < 20.524) | (positions > 42.166)
        heat.append(np.log2(np.maximum(rates / np.nanmedian(rates[outside]), 2 ** -3)))
    matrix = np.asarray(heat)
    display_matrix = smooth_rows(matrix, sigma_bins=2.0)
    image = ax.imshow(
        display_matrix,
        aspect="auto",
        interpolation="nearest",
        extent=(positions[0], positions[-1], len(rows) - 0.5, -0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
    )
    ax.axvline(20.524, color="black", lw=0.8)
    ax.axvline(42.166, color="black", lw=0.8)
    labels = [f"{row['pop']}   H={row['het_expected']:.2f}" for row in rows]
    ax.set_yticks(np.arange(len(rows)), labels)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(SPECIES[row["taxon"]])
    ax.set_xlabel("Chromosome 2L position (Mb)")
    ax.set_title("Cohort rate relative to its collinear-arm median", loc="left")
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label(r"$\log_2$ rate ratio")
    panel_label(ax, "A")

    ax = fig.add_subplot(grid[0, 1])
    for row in validation["rows"]:
        ax.scatter(
            row["het_expected"],
            row["suppression_depth"],
            s=32,
            color=SPECIES[row["taxon"]],
            edgecolor="white",
            linewidth=0.6,
        )
    h = np.asarray([row["het_expected"] for row in validation["rows"]])
    depth = np.asarray([row["suppression_depth"] for row in validation["rows"]])
    slope, intercept = np.polyfit(h, depth, 1)
    xx = np.linspace(0, 0.51, 100)
    ax.plot(xx, slope * xx + intercept, color="0.25", lw=1.1)
    ax.axhline(0, color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("Full-cohort expected 2La heterokaryotype frequency")
    ax.set_ylabel("Suppression depth $1-r_{in}/r_{out}$")
    ax.set_xlim(-0.02, 0.52)
    ax.text(
        0.03,
        0.97,
        "Pearson $r=%.2f$, $P=%.3f$\nSpearman $r_s=%.2f$, $P=%.3f$\n$n=%d$ cohorts"
        % (
            validation["pearson_Hexp_depth"][0],
            validation["pearson_Hexp_depth"][1],
            validation["spearman_Hexp_depth"][0],
            validation["spearman_Hexp_depth"][1],
            len(validation["rows"]),
        ),
        transform=ax.transAxes,
        va="top",
        size=7.2,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=color, label=f"$A.$ {name}")
        for name, color in SPECIES.items()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.8, loc="lower right")
    panel_label(ax, "B")
    save(fig, "fig3_inversion_validation")


def _nsl_ratios(nsl: dict, key: str) -> list[tuple[str, float]]:
    return [(row["species"], row[key]["ratio"]) for row in nsl["rows"]]


def _control_ratios(summary: dict) -> list[tuple[str, float]]:
    return [(row["species"], row["ratio_vs_control"]) for row in summary["rows"]]


def _sensitivity_summary(panel: dict, hap: dict, nsl: dict) -> list:
    return [
        ("Position", panel["nulls"]["position_all_arms"], _control_ratios(panel["nulls"]["position_all_arms"])),
        ("Arm + pos.", panel["nulls"]["same_arm_position"], _control_ratios(panel["nulls"]["same_arm_position"])),
        ("Core out", panel["flank_only"], _control_ratios(panel["flank_only"])),
        ("Annulus", panel["annuli"]["annulus_0p25_0p75"], _control_ratios(panel["annuli"]["annulus_0p25_0p75"])),
        ("Consensus", panel["cross_cohort_consensus"], _control_ratios(panel["cross_cohort_consensus"])),
        (
            "H12",
            hap,
            [(row["species"], row["ratio_vs_haplotype_matched"]) for row in hap["rows"]],
        ),
        ("nSL", nsl["summary"]["telo_nsl_all"], _nsl_ratios(nsl, "telo_nsl_all")),
    ]


def _bootstrap_median_ci(values: list[float], seed: int) -> tuple[float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, size=(4000, len(values)), replace=True), axis=1)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def figure4(hap: dict, panel: dict, nsl: dict) -> None:
    """Joint analysis of the frozen 15-region insecticide-resistance panel."""

    loci = list(hap["loci"])
    if loci != list(panel["loci"]) or loci != list(nsl["loci"]):
        raise ValueError("Figure 4 requires the same ordered 15-region panel in every input")
    if len(loci) != 15:
        raise ValueError(f"Figure 4 requires 15 resistance regions, found {len(loci)}")

    rows = sorted(
        hap["rows"],
        key=lambda row: (row["species"] == "arabiensis", row["cohort"]),
    )
    short_loci = [
        locus.replace("Cyp6aa/Cyp6p", "Cyp6aa/p").replace("D7r2/D7r4", "D7r2/r4")
        for locus in loci
    ]

    fig = plt.figure(figsize=(7.15, 5.15))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.48, 1.00],
        hspace=0.62,
    )
    top = outer[0].subgridspec(
        1,
        3,
        width_ratios=[5.20, 0.18, 1.05],
        wspace=0.20,
    )
    bottom = outer[1].subgridspec(
        1,
        3,
        width_ratios=[0.90, 1.28, 1.00],
        wspace=0.54,
    )

    # The main panel shows every region-population comparison rather than only
    # two group medians, so heterogeneity around parity remains visible.
    ax = fig.add_subplot(top[0, 0])
    log2_ratios = np.asarray(
        [
            [
                np.log2(row["loci"][locus]["ratio_vs_haplotype_matched"])
                for row in rows
            ]
            for locus in loci
        ]
    )
    cold_warm = LinearSegmentedColormap.from_list(
        "cold_equal_warm",
        ["#2166AC", "#F7F7F7", "#B35806"],
    )
    image = ax.imshow(
        log2_ratios,
        aspect="auto",
        interpolation="nearest",
        cmap=cold_warm,
        norm=TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=2.0),
    )
    ax.set_xticks(
        np.arange(len(rows)),
        [row["cohort"] for row in rows],
        rotation=55,
        ha="right",
    )
    ax.set_yticks(np.arange(len(loci)), short_loci)
    ax.axvline(9.5, color="0.15", lw=1.1)
    ax.set_title("Resistance-region rates are lower on average than controls", loc="left")
    panel_label(ax, "a")

    color_axis = fig.add_subplot(top[0, 1])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_ticks([-4, -2, 0, 2])
    colorbar.set_ticklabels(["≤−4", "−2", "0", "2"])
    color_axis.set_title(r"$\log_2$" "\nratio", fontsize=6.2, pad=4)

    count_axis = fig.add_subplot(top[0, 2], sharey=ax)
    cold_counts = np.sum(log2_ratios < 0, axis=1)
    y = np.arange(len(loci))
    count_axis.barh(y, cold_counts, color=BLUE, height=0.64)
    for yi, count in zip(y, cold_counts):
        place_inside = count >= 11
        count_axis.text(
            count - 0.25 if place_inside else count + 0.30,
            yi,
            f"{count}/13",
            ha="right" if place_inside else "left",
            va="center",
            fontsize=5.7,
            color="white" if place_inside else "black",
        )
    count_axis.set_xlim(0, 13)
    count_axis.set_xticks([0, 5, 10, 13])
    count_axis.tick_params(axis="y", left=False, labelleft=False)
    count_axis.set_xlabel("populations with\nratio < 1")
    count_axis.set_title("Direction across populations", loc="left", fontsize=6.6)
    count_axis.grid(axis="x", color="0.90", lw=0.5)

    ax = fig.add_subplot(bottom[0, 0])
    ratios = np.asarray([row["ratio_vs_haplotype_matched"] for row in hap["rows"]])
    is_ar = np.asarray([row["species"] == "arabiensis" for row in hap["rows"]])
    observed = np.mean(np.log(ratios[is_ar])) - np.mean(np.log(ratios[~is_ar]))
    null = []
    for indices in itertools.combinations(range(len(ratios)), int(is_ar.sum())):
        mask = np.zeros(len(ratios), bool)
        mask[list(indices)] = True
        null.append(np.mean(np.log(ratios[mask])) - np.mean(np.log(ratios[~mask])))
    ax.hist(null, bins=18, color="0.8", edgecolor="white", linewidth=0.4)
    ax.axvline(observed, color=ORANGE, lw=2)
    pvalue = hap["group_contrast"]["exact_label_permutation_p_one_sided"]
    ax.text(
        0.04,
        0.96,
        f"observed={observed:.2f}\nexact $P={pvalue:.4f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.set_xlabel("Species-group difference\nin mean log ratio")
    ax.set_ylabel("Label permutations")
    ax.set_title("exploratory group contrast", loc="left")
    panel_label(ax, "b")

    ax = fig.add_subplot(bottom[0, 1])
    summaries = _sensitivity_summary(panel, hap, nsl)
    x = np.arange(len(summaries))
    gc = np.asarray([entry[1]["median_ratio_gambcolu"] for entry in summaries])
    ar = np.asarray([entry[1]["median_ratio_arabiensis"] for entry in summaries])
    gc_ci = np.asarray(
        [
            _bootstrap_median_ci([value for species, value in entry[2] if species != "arabiensis"], 100 + index)
            for index, entry in enumerate(summaries)
        ]
    )
    ar_ci = np.asarray(
        [
            _bootstrap_median_ci([value for species, value in entry[2] if species == "arabiensis"], 200 + index)
            for index, entry in enumerate(summaries)
        ]
    )
    ax.errorbar(
        x,
        gc,
        yerr=np.vstack((gc - gc_ci[:, 0], gc_ci[:, 1] - gc)),
        color=BLUE,
        marker="o",
        ms=4,
        lw=1.2,
        capsize=2,
        label="$A.$ gambiae and $A.$ coluzzii",
    )
    ax.errorbar(
        x,
        ar,
        yerr=np.vstack((ar - ar_ci[:, 0], ar_ci[:, 1] - ar)),
        color=ORANGE,
        marker="o",
        ms=4,
        lw=1.2,
        capsize=2,
        label="$A.$ arabiensis",
    )
    ax.axhline(1, color="0.35", lw=1)
    ax.set_xticks(x, [entry[0] for entry in summaries], rotation=58, ha="right")
    ax.set_ylabel("median focal / control rate")
    upper = max(1.08, float(np.max(ar_ci[:, 1])) * 1.06)
    ax.set_ylim(0, upper)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right")
    ax.set_title("robust across control designs", loc="left")
    panel_label(ax, "c")

    ax = fig.add_subplot(bottom[0, 2])
    all_target = []
    all_control = []
    for row in nsl["rows"]:
        for locus in loci:
            record = row["loci"][locus]
            all_target.append(record["target_nsl"])
            all_control.append(record["telo_nsl_all"]["ctrl_median_nsl"])
            ax.scatter(
                record["target_nsl"], record["telo_nsl_all"]["ctrl_median_nsl"],
                color=SPECIES[row["species"]], s=9, alpha=0.72,
            )
    limit = [min(all_target + all_control) * 0.92, max(all_target + all_control) * 1.05]
    ax.plot(limit, limit, color="0.55", ls="--", lw=0.8)
    ax.set_xlim(limit)
    ax.set_ylim(limit)
    ax.set_xlabel("focal-locus nSL summary")
    ax.set_ylabel("matched-control nSL summary")
    ax.set_title("nSL-matched control balance", loc="left")
    panel_label(ax, "d")
    save(fig, "fig4_resistance_regions")


def figure_anopheles(
    validation: dict,
    resistance_panel: dict,
    hap: dict,
    nsl: dict,
) -> None:
    """One hierarchical main-text figure for the complete Anopheles result."""

    manifest = load_manifest()
    data = {cohort: load_bed(cohort) for cohort in manifest}
    offsets = arm_offsets()
    loci = list(hap["loci"])
    if (
        loci != list(resistance_panel["loci"])
        or loci != list(nsl["loci"])
        or len(loci) != 15
    ):
        raise ValueError("Combined Anopheles figure requires one ordered 15-region panel")

    rows = sorted(
        hap["rows"],
        key=lambda row: (row["species"] == "arabiensis", row["cohort"]),
    )
    short_loci = [
        locus.replace("Cyp6aa/Cyp6p", "Cyp6aa/p").replace("D7r2/D7r4", "D7r2/r4")
        for locus in loci
    ]

    fig = plt.figure(figsize=(7.15, 7.75))
    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.78, 1.26, 1.48, 0.92],
        hspace=0.58,
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.985,
    )

    # a | Genome-wide context. This is the visual entry point and therefore
    # spans the full width. The label lane is separated from the rate tracks so
    # resistance-region names never collide with data.
    atlas = outer[0].subgridspec(
        4,
        1,
        height_ratios=[0.63, 1.0, 1.0, 1.0],
        hspace=0.06,
    )
    label_axis = fig.add_subplot(atlas[0])
    rate_axes = [fig.add_subplot(atlas[index]) for index in range(1, 4)]
    total = sum(ARM_LENGTH_MB.values())
    label_axis.set_xlim(0, total)
    label_axis.set_ylim(0, 1)
    label_axis.set_axis_off()
    label_axis.set_title(
        "Recombination maps for five chromosome arms",
        loc="left",
        y=1.08,
        pad=3,
    )
    panel_label(label_axis, "a")

    species_medians = {}
    for index, (species_long, (species_short, color)) in enumerate(SPECIES_LONG.items()):
        ax = rate_axes[index]
        cohorts = [c for c, row in manifest.items() if row["species"] == species_long]
        for cohort in cohorts:
            for arm in ARMS:
                values = data[cohort][arm]
                ax.plot(
                    offsets[arm] + values[:, 0],
                    values[:, 1],
                    color=color,
                    alpha=0.17,
                    lw=0.32,
                )
        for arm in ARMS:
            arrays = [data[cohort][arm][:, 1] for cohort in cohorts]
            length = min(map(len, arrays))
            position = data[cohorts[0]][arm][:length, 0]
            median = np.nanmedian(
                np.asarray([array[:length] for array in arrays]),
                axis=0,
            )
            species_medians[(species_short, arm)] = (position, median)
            ax.plot(offsets[arm] + position, median, color=color, lw=0.95)
        ax.set_ylabel(f"$A.$ {species_short}", color=color, fontsize=5.9, labelpad=4)
        ax.set_xlim(0, total)
        ax.set_ylim(0, 12)
        ax.grid(axis="y", color="0.92", lw=0.45)
        ax.tick_params(axis="x", labelbottom=index == 2)

    for ax in rate_axes:
        for boundary in (
            offsets["2L"] + 20.524,
            offsets["2L"] + 42.166,
            offsets["2R"] + 19.024,
            offsets["2R"] + 26.759,
        ):
            ax.axvline(
                boundary,
                color="0.58",
                lw=0.55,
                ls=(0, (2.2, 2.2)),
                zorder=-1,
            )
        for arm in ARMS[1:]:
            ax.axvline(offsets[arm], color="0.84", lw=0.5, zorder=-1)

    definition_by_locus = {
        record["locus"]: record for record in resistance_panel["definitions"]
    }
    label_transform = mtransforms.blended_transform_factory(
        label_axis.transData,
        label_axis.transAxes,
    )
    label_lane_right = [-np.inf, -np.inf, -np.inf]
    label_y = [0.08, 0.43, 0.78]
    ordered_loci = sorted(
        definition_by_locus.items(),
        key=lambda item: offsets[item[1]["arm"]] + float(item[1]["mb"]),
    )
    for locus, record in ordered_loci:
        arm = record["arm"]
        mb = float(record["mb"])
        genome_x = offsets[arm] + mb
        for (species_short, _color), ax in zip(SPECIES_LONG.values(), rate_axes):
            position, median = species_medians[(species_short, arm)]
            local = np.abs(position - mb) <= 0.15
            local_rate = (
                float(np.nanmedian(median[local]))
                if np.any(local)
                else float(median[int(np.nanargmin(np.abs(position - mb)))])
            )
            ax.axvline(
                genome_x,
                color=LOCUS_PURPLE,
                alpha=0.22,
                lw=0.5,
                zorder=-1,
            )
            ax.scatter(
                genome_x,
                min(local_rate, 11.65),
                marker="^",
                s=25,
                facecolor=LOCUS_PURPLE,
                edgecolor="white",
                linewidth=0.55,
                zorder=6,
                clip_on=False,
            )
        short_label = LOCUS_SHORT.get(locus, locus)
        half_width = max(2.5, 0.58 * len(short_label))
        label_x = float(np.clip(genome_x, half_width, total - half_width))
        lane = next(
            (
                lane_index
                for lane_index, previous_right in enumerate(label_lane_right)
                if label_x - half_width >= previous_right + 1.0
            ),
            int(np.argmin(label_lane_right)),
        )
        label_lane_right[lane] = label_x + half_width
        label_axis.plot(
            [genome_x, label_x],
            [0, label_y[lane] - 0.05],
            color=LOCUS_PURPLE,
            alpha=0.30,
            lw=0.4,
            clip_on=False,
        )
        label_axis.text(
            label_x,
            label_y[lane],
            short_label,
            transform=label_transform,
            ha="center",
            va="bottom",
            color=LOCUS_PURPLE,
            fontsize=4.8,
            fontweight="bold",
            clip_on=False,
        )

    centers = [offsets[arm] + ARM_LENGTH_MB[arm] / 2 for arm in ARMS]
    rate_axes[-1].set_xticks(centers, ARMS)
    rate_axes[-1].set_xlabel("Chromosome arm")
    rate_axes[0].text(
        offsets["2L"] + 31.3,
        11.3,
        "2La",
        ha="center",
        color="0.35",
    )
    rate_axes[0].text(
        offsets["2R"] + 22.9,
        11.3,
        "2Rb",
        ha="center",
        color="0.4",
    )

    # b-c | The established 2La inversion supplies the positive-control
    # prediction before the resistance-region result.
    validation_row = outer[1].subgridspec(
        1,
        2,
        width_ratios=[1.34, 1.0],
        wspace=0.47,
    )
    validation_rows = sorted(validation["rows"], key=lambda row: row["het_expected"])
    ax_heat = fig.add_subplot(validation_row[0, 0])
    heat = []
    positions = None
    for row in validation_rows:
        values = data[row["pop"]]["2L"]
        positions = values[:, 0]
        rates = values[:, 1]
        outside = (positions < 20.524) | (positions > 42.166)
        heat.append(np.log2(np.maximum(rates / np.nanmedian(rates[outside]), 2 ** -3)))
    matrix = np.asarray(heat)
    image = ax_heat.imshow(
        smooth_rows(matrix, sigma_bins=2.0),
        aspect="auto",
        interpolation="nearest",
        extent=(positions[0], positions[-1], len(validation_rows) - 0.5, -0.5),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0),
    )
    ax_heat.axvline(20.524, color="black", lw=0.65)
    ax_heat.axvline(42.166, color="black", lw=0.65)
    ax_heat.set_yticks(
        np.arange(len(validation_rows)),
        [f"{row['pop']}  H={row['het_expected']:.2f}" for row in validation_rows],
    )
    for tick, row in zip(ax_heat.get_yticklabels(), validation_rows):
        tick.set_color(SPECIES[row["taxon"]])
    ax_heat.set_xlabel("Chromosome 2L position (Mb)")
    ax_heat.set_title("Cold block depth varies between cohorts", loc="left")
    colorbar = fig.colorbar(image, ax=ax_heat, fraction=0.035, pad=0.02)
    colorbar.set_label(r"$\log_2$ rate ratio")
    panel_label(ax_heat, "b")

    ax_prediction = fig.add_subplot(validation_row[0, 1])
    for row in validation["rows"]:
        ax_prediction.scatter(
            row["het_expected"],
            row["suppression_depth"],
            s=24,
            color=SPECIES[row["taxon"]],
            edgecolor="white",
            linewidth=0.5,
        )
    h = np.asarray([row["het_expected"] for row in validation["rows"]])
    depth = np.asarray([row["suppression_depth"] for row in validation["rows"]])
    slope, intercept = np.polyfit(h, depth, 1)
    xx = np.linspace(0, 0.51, 100)
    ax_prediction.plot(xx, slope * xx + intercept, color="0.25", lw=1.0)
    ax_prediction.axhline(0, color="0.6", ls="--", lw=0.7)
    ax_prediction.set_xlabel("Expected 2La heterokaryotype frequency")
    ax_prediction.set_ylabel(r"Suppression depth $1-r_{\rm in}/r_{\rm out}$")
    ax_prediction.set_xlim(-0.02, 0.52)
    ax_prediction.text(
        0.03,
        0.97,
        "Pearson $r=%.2f$, $P=%.3f$\nSpearman $r_s=%.2f$, $P=%.3f$\n$n=%d$ cohorts"
        % (
            validation["pearson_Hexp_depth"][0],
            validation["pearson_Hexp_depth"][1],
            validation["spearman_Hexp_depth"][0],
            validation["spearman_Hexp_depth"][1],
            len(validation["rows"]),
        ),
        transform=ax_prediction.transAxes,
        va="top",
        size=6.2,
    )
    ax_prediction.set_title("Suppression matches mixture model prediction", loc="left")
    panel_label(ax_prediction, "c")

    # d | The biological outcome receives its own full-width row. The marginal
    # counts are part of the same panel because they summarize the heat-map rows.
    resistance = outer[2].subgridspec(
        1,
        3,
        width_ratios=[5.2, 0.18, 1.05],
        wspace=0.20,
    )
    ax_resistance = fig.add_subplot(resistance[0, 0])
    log2_ratios = np.asarray(
        [
            [
                np.log2(row["loci"][locus]["ratio_vs_haplotype_matched"])
                for row in rows
            ]
            for locus in loci
        ]
    )
    cold_warm = LinearSegmentedColormap.from_list(
        "cold_equal_warm_editorial",
        [BLUE, "#FAFAF8", GRAY],
    )
    resistance_image = ax_resistance.imshow(
        log2_ratios,
        aspect="auto",
        interpolation="nearest",
        cmap=cold_warm,
        norm=TwoSlopeNorm(vmin=-4.0, vcenter=0.0, vmax=2.0),
    )
    ax_resistance.set_xticks(
        np.arange(len(rows)),
        [row["cohort"] for row in rows],
        rotation=48,
        ha="right",
    )
    ax_resistance.set_yticks(np.arange(len(loci)), short_loci)
    ax_resistance.axvline(9.5, color="0.15", lw=1.0)
    ax_resistance.set_title(
        "Resistant region rates are lower than control",
        loc="left",
    )
    panel_label(ax_resistance, "d")

    color_axis = fig.add_subplot(resistance[0, 1])
    colorbar = fig.colorbar(resistance_image, cax=color_axis)
    colorbar.set_ticks([-4, -2, 0, 2])
    colorbar.set_ticklabels(["≤−4", "−2", "0", "2"])
    color_axis.set_title(r"$\log_2$" "\nratio", fontsize=6.0, pad=3)

    count_axis = fig.add_subplot(resistance[0, 2], sharey=ax_resistance)
    cold_counts = np.sum(log2_ratios < 0, axis=1)
    y = np.arange(len(loci))
    count_axis.barh(y, cold_counts, color=BLUE, height=0.64)
    for yi, count in zip(y, cold_counts):
        inside = count >= 11
        count_axis.text(
            count - 0.25 if inside else count + 0.30,
            yi,
            f"{count}/13",
            ha="right" if inside else "left",
            va="center",
            fontsize=5.5,
            color="white" if inside else "black",
        )
    count_axis.set_xlim(0, 13)
    count_axis.set_xticks([0, 5, 10, 13])
    count_axis.tick_params(axis="y", left=False, labelleft=False)
    count_axis.set_xlabel("Populations with\nratio < 1")
    count_axis.set_title("Direction in cohorts", loc="left", fontsize=6.5)
    count_axis.grid(axis="x", color="0.90", lw=0.5)

    # e-f | Statistical transparency and control-design robustness close the
    # story without giving QC panels the same visual weight as the result.
    controls = outer[3].subgridspec(
        1,
        2,
        width_ratios=[0.82, 1.48],
        wspace=0.50,
    )
    ax_null = fig.add_subplot(controls[0, 0])
    cohort_ratios = np.asarray(
        [row["ratio_vs_haplotype_matched"] for row in hap["rows"]]
    )
    is_arabiensis = np.asarray(
        [row["species"] == "arabiensis" for row in hap["rows"]]
    )
    observed = np.mean(np.log(cohort_ratios[is_arabiensis])) - np.mean(
        np.log(cohort_ratios[~is_arabiensis])
    )
    null = []
    for indices in itertools.combinations(
        range(len(cohort_ratios)),
        int(is_arabiensis.sum()),
    ):
        mask = np.zeros(len(cohort_ratios), bool)
        mask[list(indices)] = True
        null.append(
            np.mean(np.log(cohort_ratios[mask]))
            - np.mean(np.log(cohort_ratios[~mask]))
        )
    ax_null.hist(null, bins=18, color="0.80", edgecolor="white", linewidth=0.35)
    ax_null.axvline(observed, color=BLUE, lw=1.8)
    pvalue = hap["group_contrast"]["exact_label_permutation_p_one_sided"]
    ax_null.text(
        0.04,
        0.96,
        f"observed={observed:.2f}\nexact $P={pvalue:.4f}$",
        transform=ax_null.transAxes,
        ha="left",
        va="top",
        fontsize=6.0,
    )
    ax_null.set_xlabel("Species-group difference\nin mean log ratio")
    ax_null.set_ylabel("Label permutations")
    ax_null.set_title("Cohort analysis design", loc="left")
    panel_label(ax_null, "e")

    ax_controls = fig.add_subplot(controls[0, 1])
    summaries = _sensitivity_summary(resistance_panel, hap, nsl)
    x = np.arange(len(summaries))
    gc = np.asarray([entry[1]["median_ratio_gambcolu"] for entry in summaries])
    ar = np.asarray([entry[1]["median_ratio_arabiensis"] for entry in summaries])
    gc_ci = np.asarray(
        [
            _bootstrap_median_ci(
                [value for species, value in entry[2] if species != "arabiensis"],
                100 + index,
            )
            for index, entry in enumerate(summaries)
        ]
    )
    ar_ci = np.asarray(
        [
            _bootstrap_median_ci(
                [value for species, value in entry[2] if species == "arabiensis"],
                200 + index,
            )
            for index, entry in enumerate(summaries)
        ]
    )
    ax_controls.errorbar(
        x,
        gc,
        yerr=np.vstack((gc - gc_ci[:, 0], gc_ci[:, 1] - gc)),
        color=BLUE,
        marker="o",
        ms=3.8,
        lw=1.1,
        capsize=2,
        label="$A.$ gambiae and $A.$ coluzzii",
    )
    ax_controls.errorbar(
        x,
        ar,
        yerr=np.vstack((ar - ar_ci[:, 0], ar_ci[:, 1] - ar)),
        color=ORANGE,
        marker="o",
        ms=3.8,
        lw=1.1,
        capsize=2,
        label="$A.$ arabiensis",
    )
    ax_controls.axhline(1, color="0.35", lw=0.9)
    ax_controls.set_xticks(
        x,
        [entry[0] for entry in summaries],
        rotation=40,
        ha="right",
    )
    ax_controls.set_ylabel("Median focal / control")
    ax_controls.set_ylim(0, max(1.08, float(np.max(ar_ci[:, 1])) * 1.06))
    ax_controls.legend(frameon=False, fontsize=5.2, loc="lower right")
    ax_controls.set_title("Control analysis design", loc="left")
    ax_controls.text(
        -0.075,
        1.11,
        "f",
        transform=ax_controls.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )

    save(fig, "fig_anopheles_combined")


def main() -> None:
    style()
    requested = set(
        sys.argv[1:] or ("fig1", "fig2", "fig3", "fig4", "fig_anopheles")
    )
    unknown = requested - {"fig1", "fig2", "fig3", "fig4", "fig_anopheles"}
    if unknown:
        raise SystemExit("unknown figure target(s): " + ", ".join(sorted(unknown)))
    summary = json.loads((ROOT / "paper" / "results_snapshot" / "summary.json").read_text())
    demography = json.loads(
        (ROOT / "paper" / "results_snapshot" / "demography_matched.json").read_text()
    )
    selection = json.loads((ROOT / "paper" / "figdata" / "selection_dr.json").read_text())
    selection_windows = np.load(ROOT / "paper" / "figdata" / "selection_dr_figdata.npz")
    if "fig1" in requested:
        figure1(summary, selection, selection_windows, demography)
    if requested & {"fig2", "fig3", "fig4", "fig_anopheles"}:
        validation = json.loads(
            (ROOT / "paper" / "results_snapshot" / "agam_validation.json").read_text()
        )
        panel_data = json.loads(
            (ROOT / "paper" / "figdata" / "agam_resistance_panel_sensitivity.json").read_text()
        )
        hap = json.loads(
            (
                ROOT
                / "paper"
                / "figdata"
                / "agam_haplotype_matched_controls_hancock_mechanisms.json"
            ).read_text()
        )
        nsl = json.loads(
            (
                ROOT
                / "paper"
                / "figdata"
                / "agam_nsl_matched_controls_hancock_mechanisms.json"
            ).read_text()
        )
        if "fig2" in requested:
            figure2(validation, panel_data["panels"]["hancock_mechanisms"])
        if "fig3" in requested:
            figure3(validation)
        if "fig4" in requested:
            figure4(hap, panel_data["panels"]["hancock_mechanisms"], nsl)
        if "fig_anopheles" in requested:
            figure_anopheles(
                validation,
                panel_data["panels"]["hancock_mechanisms"],
                hap,
                nsl,
            )


if __name__ == "__main__":
    main()
