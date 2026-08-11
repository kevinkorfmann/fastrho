"""Render the frozen-checkpoint gene-conversion operating-regime test.

The figure is a pure view of two committed artifacts produced by
``scripts/gene_conversion_stress.py``.  Aggregate estimates and region-bootstrap
confidence intervals come from ``paper/results_snapshot/gene_conversion.json``;
the representative paired track comes from ``paper/figdata/gene_conversion.npz``.
No result is recomputed by inference at plotting time.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import paper_style as ps


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "paper" / "results_snapshot" / "gene_conversion.json"
ARRAYS = ROOT / "paper" / "figdata" / "gene_conversion.npz"

TRACTS = (100, 300, 1000)
COLORS = {100: ps.C["fastrho"], 300: ps.C["truth"], 1000: ps.C["relernn"]}
MARKERS = {100: "o", 300: "s", 1000: "^"}
LINESTYLES = {100: "-", 300: "--", 1000: "-."}


def _conditions_by_tract(summary: dict, tract: int) -> list[dict]:
    baseline = next(row for row in summary["conditions"] if row["id"] == "gc0")
    rows = [
        row
        for row in summary["conditions"]
        if row["tract_length"] == tract and row["ratio"] > 0
    ]
    return [baseline, *sorted(rows, key=lambda row: row["ratio"])]


def _series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([row["ratio"] for row in rows], dtype=float)
    y = np.asarray([row["pooled"][metric]["estimate"] for row in rows], dtype=float)
    ci = np.asarray([row["pooled"][metric]["ci95"] for row in rows], dtype=float)
    return x, y, ci


def _rowwise_pearson(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Pearson correlation for each paired simulated region."""
    truth_centered = truth - np.mean(truth, axis=1, keepdims=True)
    prediction_centered = prediction - np.mean(prediction, axis=1, keepdims=True)
    numerator = np.sum(truth_centered * prediction_centered, axis=1)
    denominator = np.sqrt(
        np.sum(truth_centered**2, axis=1)
        * np.sum(prediction_centered**2, axis=1)
    )
    return numerator / denominator


def _representative_region(
    summary: dict,
    truth: np.ndarray,
    baseline: np.ndarray,
    severe: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Choose the replicate nearest the three reported display targets.

    The deterministic rule jointly considers baseline shape, severe-condition
    shape, and the severe-to-baseline mean-rate ratio.  Robust scaling prevents
    any one metric from dominating because of its numerical units.
    """
    baseline_summary = next(row for row in summary["conditions"] if row["id"] == "gc0")
    severe_summary = next(
        row for row in summary["conditions"] if row["id"] == "gc4_t1000"
    )
    metrics = np.column_stack(
        [
            _rowwise_pearson(truth, baseline),
            _rowwise_pearson(truth, severe),
            np.mean(severe, axis=1) / np.mean(baseline, axis=1),
        ]
    )
    target = np.asarray(
        [
            baseline_summary["pooled"]["pearson"]["estimate"],
            severe_summary["pooled"]["pearson"]["estimate"],
            severe_summary["pooled"]["paired_mean_rate_vs_no_conversion"]["estimate"],
        ]
    )
    median = np.median(metrics, axis=0)
    mad = np.median(np.abs(metrics - median), axis=0)
    mad = np.where(mad > 0, mad, 1.0)
    score = np.sum(np.abs(metrics - target) / mad, axis=1)
    region_index = int(np.argmin(score))
    return region_index, metrics[region_index]


def _draw_summary(ax, summary: dict, metric: str, ylabel: str) -> None:
    for tract in TRACTS:
        x, y, ci = _series(_conditions_by_tract(summary, tract), metric)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack((y - ci[:, 0], ci[:, 1] - y)),
            color=COLORS[tract],
            marker=MARKERS[tract],
            linewidth=1.7,
            markersize=5.0,
            linestyle=LINESTYLES[tract],
            capsize=2.0,
            elinewidth=0.8,
            label=f"{tract:,} bp",
        )
    ax.axvline(0, color="#777777", linewidth=0.8, zorder=0)
    ax.set_xticks([0, 0.5, 1, 2, 4])
    ax.set_xlabel("Initiation-rate ratio\n(gene conversion : crossover)")
    ax.set_ylabel(ylabel)


def main() -> None:
    ps.style()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS, allow_pickle=False)

    # Match the oversized native scale of the dense input-view SI figure so that
    # LaTeX down-scaling harmonizes apparent fonts and line weights.
    fig = plt.figure(figsize=(11.0, 8.1))
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 0.92), hspace=0.58, wspace=0.34)
    ax_shape = fig.add_subplot(grid[0, 0])
    ax_scale = fig.add_subplot(grid[0, 1])
    ax_track = fig.add_subplot(grid[1, :])

    _draw_summary(
        ax_shape,
        summary,
        "pearson",
        "Pearson $r$ with crossover map",
    )
    ax_shape.set_ylim(0.6, 0.94)
    ax_shape.legend(title="Mean tract length", loc="lower left", ncol=1)
    ps.panel(ax_shape, "a", x=-0.09, y=1.08)

    _draw_summary(
        ax_scale,
        summary,
        "paired_mean_rate_vs_no_conversion",
        "Mean inferred rate /\npaired no-conversion run",
    )
    ax_scale.axhline(1, color="#555555", linewidth=1.0, linestyle=":", zorder=0)
    ax_scale.set_ylim(0.8, 3.3)
    ps.panel(ax_scale, "b", x=-0.09, y=1.08)

    ids = arrays["condition_id"]
    baseline_index = int(np.flatnonzero(ids == "gc0")[0])
    severe_index = int(np.flatnonzero(ids == "gc4_t1000")[0])
    truth = arrays["truth_25kb"]
    baseline = arrays["predicted_25kb"][baseline_index]
    severe = arrays["predicted_25kb"][severe_index]
    region_index, display_metrics = _representative_region(
        summary, truth, baseline, severe
    )
    centers_mb = (
        arrays["window_edges"][:-1] + np.diff(arrays["window_edges"]) / 2
    ) / 1e6
    truth_track = truth[region_index]
    baseline_track = baseline[region_index]
    severe_track = severe[region_index]
    truth_track = truth_track / np.mean(truth_track)
    baseline_track = baseline_track / np.mean(baseline_track)
    severe_track = severe_track / np.mean(severe_track)
    ax_track.axhline(1, color="#777777", linewidth=0.8, linestyle=":", zorder=0)
    ax_track.plot(
        centers_mb,
        truth_track,
        color=ps.C["truth"],
        linewidth=1.6,
        label="Crossover truth",
    )
    ax_track.plot(
        centers_mb,
        baseline_track,
        color=ps.C["fastrho"],
        linewidth=1.5,
        label="No gene conversion",
    )
    ax_track.plot(
        centers_mb,
        severe_track,
        color=COLORS[1000],
        linewidth=1.5,
        linestyle=LINESTYLES[1000],
        label="4× initiation, 1,000-bp tracts",
    )
    ax_track.set_xlabel("Position (Mb)")
    ax_track.set_ylabel("Mean-normalized rate")
    ax_track.set_xlim(0, 1)
    ax_track.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        fontsize=8.2,
    )
    ax_track.text(
        0.01,
        0.95,
        (
            f"Representative replicate: r = {display_metrics[0]:.2f} (no conversion), "
            f"{display_metrics[1]:.2f} (severe); "
            f"raw mean shift = {display_metrics[2]:.2f}×"
        ),
        transform=ax_track.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
    )
    ps.panel(ax_track, "c", x=-0.08)

    fig.align_ylabels([ax_shape, ax_scale])
    ps.save(fig, "fig_gene_conversion", formats=("pdf", "png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
