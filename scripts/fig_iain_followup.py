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
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7B3294"
GRAY = "#6B6B6B"
LIGHT = "#D8D8D8"


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
    for species, color, label in (("wolf", BLUE, "wolf"), ("dog", ORANGE, "village dog")):
        values = np.array([
            [r["value"] for r in rows if r["metric"] == "pearson"
             and r["species"] == species and r["window_bp"] == scale]
            for scale in scales
        ])
        for replicate in range(values.shape[1]):
            ax.plot(x, values[:, replicate], color=color, alpha=0.10, lw=0.55, zorder=1)
        median = np.median(values, axis=1)
        lo, hi = np.quantile(values, [0.025, 0.975], axis=1)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(x, median, color=color, marker="o", ms=4.2, lw=1.8, label=label, zorder=3)
    ax.set_xticks(x, ["100 kb", "200 kb", "500 kb", "1 Mb"])
    ax.set_ylabel("Pearson correlation\nwith pedigree map")
    ax.set_title("a  Matched sample size, local density, and MAF", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right")


def panel_canid_chromosomes(ax, data: dict) -> None:
    level = data["scales"]["100000"]["log_pearson"]
    wolf = np.asarray(level["wolf_per_chromosome"])
    dog = np.asarray(level["dog_per_chromosome"])
    x = np.arange(1, len(wolf) + 1)
    for index in range(len(x)):
        ax.plot([x[index] - 0.08, x[index] + 0.08], [dog[index], wolf[index]],
                color=LIGHT, lw=1.4, zorder=1)
    ax.scatter(x - 0.08, wolf, s=27, color=BLUE, label="wolf", zorder=3)
    ax.scatter(x + 0.08, dog, s=27, color=ORANGE, label="village dog", zorder=3)
    delta = level["wolf_minus_dog_mean"]
    lo, hi = level["wolf_minus_dog_ci95_chromosome_bootstrap"]
    ax.text(0.42, 0.96, f"mean difference = {delta:.3f}\n95% CI {lo:.3f} to {hi:.3f}",
            transform=ax.transAxes, va="top", fontsize=7.5)
    ax.set_xticks(x, [f"chr {i}" for i in x])
    ax.set_ylabel("Log-rate Pearson correlation\nwith pedigree map (100 kb)")
    ax.set_title("b  Empirical canid replication", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="lower right")


def panel_biology(ax, data: dict) -> None:
    maps = ["Campbell pedigree", "wolf LD", "dog LD"]
    covariates = [
        ("partial_beta_gc", "GC", GREEN),
        ("partial_beta_log1p_tss", "TSS density", PURPLE),
        ("partial_beta_log1p_snp_density", "SNP density", GRAY),
    ]
    base = np.arange(len(maps))
    offsets = [-0.22, 0.0, 0.22]
    for offset, (key, label, color) in zip(offsets, covariates):
        estimates, lower, upper = [], [], []
        for name in maps:
            item = data["maps"][name][key]
            estimates.append(item["estimate"])
            lower.append(item["ci95_5mb_block_bootstrap"][0])
            upper.append(item["ci95_5mb_block_bootstrap"][1])
        estimates = np.asarray(estimates)
        errors = np.vstack((estimates - np.asarray(lower), np.asarray(upper) - estimates))
        ax.errorbar(base + offset, estimates, yerr=errors, fmt="o", ms=4, capsize=2,
                    lw=1.1, color=color, label=label)
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_xticks(base, ["Pedigree", "Wolf LD", "Dog LD"])
    ax.set_ylabel("Adjusted standardized coefficient")
    ax.set_title("c  Canid biological and ascertainment signals", loc="left", fontweight="bold")
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
    colors = [GREEN, PURPLE, GRAY]
    rng = np.random.default_rng(20260814)
    for index, ((label, values, ci, mean), color) in enumerate(zip(entries, colors)):
        jitter = rng.uniform(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, color=color, s=22,
                   alpha=0.80, zorder=2)
        ax.errorbar(index, mean, yerr=[[mean - ci[0]], [ci[1] - mean]], fmt="D",
                    color="#111111", mfc="white", ms=4.5, capsize=3, lw=1.2, zorder=3)
    ax.set_xticks(range(len(entries)), [entry[0] for entry in entries])
    ax.set_ylabel("Chromosome-level correlation (100 kb)")
    ax.set_title("d  Multi-chromosome portability follow-up", loc="left", fontweight="bold")
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

    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), constrained_layout=True)
    panel_marker_match(axes[0, 0], rows)
    panel_canid_chromosomes(axes[0, 1], multichrom)
    panel_biology(axes[1, 0], biology)
    panel_cross_species(axes[1, 1], dmel, jewel, aspen)
    for ax in axes.ravel():
        ax.grid(axis="y", color="#E8E8E8", lw=0.6, zorder=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
