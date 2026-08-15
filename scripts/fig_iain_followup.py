"""Plot the analyses added in response to Iain Mathieson's review.

The four panels distinguish external-map accuracy from split-panel repeatability
and keep the strict complete-call marker match as the primary canid sensitivity.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps


COBALT = ps.C["fastrho"]
INK = ps.C["truth"]
GRAY = ps.C["relernn"]
LIGHT_GRAY = "#D9D9D6"
PALE_COBALT = ps.C["fastrho_l"]


def panel_heading(ax, letter: str, title: str) -> None:
    """Use the manuscript convention: bold panel letter, regular title."""
    ps.panel(ax, letter, x=-0.12, y=1.075, fontsize=10.5)
    ax.set_title(title, loc="left", pad=7, fontweight="normal")


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_long(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["replicate"] = int(row["replicate"])
        row["window_bp"] = int(row["window_bp"])
        row["value"] = float(row["value"])
    return rows


def panel_marker_match(ax, rows: list[dict]) -> None:
    scales = [100_000, 200_000, 500_000, 1_000_000]
    x = np.arange(len(scales))
    series = (
        ("wolf", COBALT, "o", COBALT, "wolf"),
        ("dog", GRAY, "^", "white", "village dog"),
    )
    for species, color, marker, marker_face, label in series:
        values = np.array([
            [r["value"] for r in rows if r["metric"] == "pearson"
             and r["species"] == species and r["window_bp"] == scale]
            for scale in scales
        ])
        for replicate in range(values.shape[1]):
            ax.plot(x, values[:, replicate], color=color, alpha=0.10, lw=0.55, zorder=1)
        median = np.median(values, axis=1)
        lo, hi = np.quantile(values, [0.025, 0.975], axis=1)
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot(
            x,
            median,
            color=color,
            marker=marker,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=1.0,
            ms=4.8,
            lw=1.8,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, ["100 kb", "200 kb", "500 kb", "1 Mb"])
    ax.set_ylabel("Pearson correlation\nwith pedigree map")
    panel_heading(ax, "a", "Matched sample size, marker density, and MAF")
    ax.legend(frameon=False, ncol=2, loc="lower right")


def panel_canid_chromosomes(ax, data: dict) -> None:
    level = data["scales"]["100000"]["log_pearson"]
    wolf = np.asarray(level["wolf_per_chromosome"])
    dog = np.asarray(level["dog_per_chromosome"])
    y = np.arange(1, len(wolf) + 1)
    for index in range(len(y)):
        ax.plot([dog[index], wolf[index]], [y[index], y[index]],
                color=LIGHT_GRAY, lw=1.8, zorder=1)
    ax.scatter(wolf, y, s=30, color=COBALT, marker="o", label="wolf", zorder=3)
    ax.scatter(
        dog,
        y,
        s=34,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=1.2,
        marker="^",
        label="village dog",
        zorder=3,
    )
    delta = level["wolf_minus_dog_mean"]
    lo, hi = level["wolf_minus_dog_ci95_chromosome_bootstrap"]
    ax.text(
        0.98,
        0.04,
        f"mean wolf - dog = {delta:.3f}\n95% CI {lo:.3f} to {hi:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#333333",
    )
    ax.set_yticks(y, [f"chr {i}" for i in y])
    ax.invert_yaxis()
    ax.set_xticks([0.20, 0.25, 0.30, 0.35])
    ax.set_xlabel("Log-rate correlation with pedigree map (100 kb)")
    panel_heading(ax, "b", "Five-chromosome canid replication")
    ax.legend(frameon=False, ncol=1, loc="upper left")


def panel_biology(ax, data: dict) -> None:
    maps = ["Campbell pedigree", "wolf LD", "dog LD"]
    covariates = [
        ("partial_beta_gc", "GC", COBALT, "o"),
        ("partial_beta_log1p_tss", "TSS density", INK, "s"),
        ("partial_beta_log1p_snp_density", "SNP density", GRAY, "D"),
    ]
    base = np.arange(len(maps))
    offsets = [-0.22, 0.0, 0.22]
    for offset, (key, label, color, marker) in zip(offsets, covariates):
        estimates, lower, upper = [], [], []
        for name in maps:
            item = data["maps"][name][key]
            estimates.append(item["estimate"])
            lower.append(item["ci95_5mb_block_bootstrap"][0])
            upper.append(item["ci95_5mb_block_bootstrap"][1])
        estimates = np.asarray(estimates)
        errors = np.vstack((estimates - np.asarray(lower), np.asarray(upper) - estimates))
        ax.errorbar(base + offset, estimates, yerr=errors, fmt=marker, ms=4.2, capsize=2,
                    lw=1.1, color=color, label=label)
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(base, ["Pedigree", "Wolf LD", "Dog LD"])
    ax.set_ylabel("Adjusted standardized coefficient")
    panel_heading(ax, "c", "Biological and ascertainment signals")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")


def _species_level(payload: dict, comparison: str, metric: str = "pearson") -> tuple[np.ndarray, tuple[float, float], float]:
    level = payload["aggregate"]["scales"]["100000"][comparison][metric]
    return (np.asarray(level["per_chromosome"], dtype=float),
            tuple(level["ci95_chromosome_bootstrap"]), float(level["mean"]))


def panel_cross_species(ax, dmel: dict, jewel: dict, aspen: dict) -> None:
    entries = [
        ("Fruit fly\nexternal",) + _species_level(dmel, "external_map", "pearson"),
        ("Jewel wasp\nsplit sample",) + _species_level(jewel, "split_reproducibility", "spearman"),
        ("Aspen\nsplit sample",) + _species_level(aspen, "split_reproducibility", "spearman"),
    ]
    colors = [INK, GRAY, GRAY]
    markers = ["o", "s", "^"]
    rng = np.random.default_rng(20260814)
    for index, ((label, values, ci, mean), color, marker) in enumerate(
        zip(entries, colors, markers)
    ):
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            facecolor=color if index == 0 else "white",
            edgecolor=color,
            linewidth=0.9,
            marker=marker,
            s=24,
            alpha=0.90,
            zorder=2,
        )
        ax.errorbar(index, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="D",
                    color=COBALT, mfc="white", mec=COBALT, mew=1.1,
                    ms=4.8, capsize=3, lw=1.2, zorder=3)
    ax.set_xticks(range(len(entries)), [entry[0] for entry in entries])
    ax.set_ylabel("Chromosome-level correlation (100 kb)")
    panel_heading(ax, "d", "Cross-chromosome portability")
    ax.text(0.02, 0.97, "Pearson: external map; Spearman: split-sample repeatability",
            transform=ax.transAxes, va="top", fontsize=7, color="#333333")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_dir

    rows = load_long(root / "canid_marker_matched_complete_long.csv")
    multichrom = load_json(root / "canid_multichrom.json")
    biology = load_json(root / "canid_biological_validation_adjusted.json")
    dmel = load_json(root / "dmel_multichrom.json")
    jewel = load_json(root / "jewelwasp_multichrom.json")
    aspen = load_json(root / "aspen_multichrom.json")

    ps.style()
    mpl.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9.0,
        "axes.titleweight": "normal",
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.55), constrained_layout=True)
    panel_marker_match(axes[0, 0], rows)
    panel_canid_chromosomes(axes[0, 1], multichrom)
    panel_biology(axes[1, 0], biology)
    panel_cross_species(axes[1, 1], dmel, jewel, aspen)
    for ax in axes.ravel():
        ax.grid(axis="y", color="#D4D4D0", alpha=0.35, lw=0.45, zorder=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
