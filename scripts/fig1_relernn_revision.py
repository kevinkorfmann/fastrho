#!/usr/bin/env python3
"""Render the isolated reviewer-response version of main-text Figure 1.

Panel A deliberately separates the fine-scale 25-kb comparison from a
prespecified constant-rate test in ReLERNN's intended native-window regime.
The timing panel reports measured workflow stages rather than a normalized
cross-hardware speed ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2737E7"
BLACK = "#151515"
GRAY = "#777777"
SCENARIOS = [
    ("Const.", "const_n20", "constant"),
    ("Btl.", "bottleneck_n20", "bottleneck"),
    ("Exp.", "expansion_n20", "expansion"),
    ("deC.", "real_decode", "decode"),
    ("Hap.", "real_hapmap", "hapmap"),
    ("Dog", "real_dog", "dog"),
]
METHODS = (
    ("fastrho", "fastrho", BLUE, "o"),
    ("pyrho", "pyrho", BLACK, "s"),
    ("relernn", "ReLERNN", GRAY, "^"),
)


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
            "axes.labelcolor": BLACK,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 150,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(-0.11, 1.11, label, transform=axis.transAxes, weight="bold", size=9)


def paired_record(demography: dict, key: str, method: str, history: str) -> dict:
    scenario = key.removesuffix("_n20")
    if method == "fastrho":
        return demography["scenarios"][scenario]["fastrho_reference"]["25kb"]
    return demography["scenarios"][scenario][method]["arms"][history]["25kb"]


def fine_scale_record(summary: dict, demography: dict, key: str, method: str) -> dict:
    if key in {"bottleneck_n20", "expansion_n20"}:
        return paired_record(demography, key, method, "matched")
    return summary[key]["scales"]["25kb"][method]


def render(
    summary: dict,
    demography: dict,
    native: dict,
    misspecified: dict,
    selection: dict,
    selection_windows: np.lib.npyio.NpzFile,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.15, 4.28))
    outer = fig.add_gridspec(2, 1, hspace=0.78)
    top = outer[0].subgridspec(1, 3, width_ratios=[1.55, 1.00, 1.25], wspace=0.54)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[0.95, 1.30], wspace=0.46)

    # a | Same six scenarios at two scientifically distinct resolution regimes.
    accuracy = top[0, 0].subgridspec(2, 1, hspace=0.38)
    ax_fine = fig.add_subplot(accuracy[0, 0])
    ax_native = fig.add_subplot(accuracy[1, 0])
    x = np.arange(len(SCENARIOS))
    for method, label, color, marker in METHODS[:2]:
        values = [
            fine_scale_record(summary, demography, key, method)["pearson"]
            for _short, key, _native_key in SCENARIOS
        ]
        ax_fine.plot(x, values, color=color, marker=marker, ms=3.8, lw=1.05)
        ax_fine.text(
            x[-1] + 0.10,
            values[-1] + (0.02 if method == "fastrho" else -0.025),
            label,
            color=color,
            fontsize=5.5,
            va="center",
            clip_on=False,
        )
        if method == "pyrho":
            for index, (_short, key, _native_key) in enumerate(SCENARIOS[1:3], start=1):
                constant = paired_record(demography, key, method, "constant")["pearson"]
                matched = paired_record(demography, key, method, "matched")["pearson"]
                ax_fine.plot([index - 0.08, index], [constant, matched], color=color, lw=0.6, ls=":")
                ax_fine.scatter(
                    index - 0.08,
                    constant,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    s=16,
                    linewidth=0.8,
                    zorder=4,
                )
    ax_fine.set_xticks(x, [""] * len(SCENARIOS))
    ax_fine.set_xlim(-0.25, len(SCENARIOS) - 0.02)
    ax_fine.set_ylim(0, 1.08)
    ax_fine.set_ylabel("Pearson $r$")
    ax_fine.grid(axis="y", color="0.90", lw=0.5)
    ax_fine.set_title("Fine-scale benchmark (25 kb)", loc="left", pad=4)
    ax_fine.legend(
        handles=[
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black", linestyle="none", markersize=3.8, label="constant"),
            Line2D([], [], marker="o", markerfacecolor="black", markeredgecolor="black", linestyle="none", markersize=3.8, label="matched"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.01, 1.01),
        ncol=2,
        frameon=False,
        fontsize=4.8,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0,
    )
    panel_label(ax_fine, "a")

    for method, label, color, marker in METHODS:
        values = [native["scenarios"][native_key]["methods"][method]["pearson"] for _short, _key, native_key in SCENARIOS]
        ax_native.plot(x, values, color=color, marker=marker, ms=3.8, lw=1.05, zorder=3)
        ax_native.text(
            x[-1] + 0.10,
            values[-1] + {"fastrho": 0.025, "pyrho": -0.018, "relernn": -0.075}[method],
            label,
            color=color,
            fontsize=5.4,
            va="center",
            clip_on=False,
        )
        if method in {"pyrho", "relernn"}:
            for index, native_key in ((1, "bottleneck"), (2, "expansion")):
                constant = misspecified["scenarios"][native_key]["methods"][method]["pearson"]
                matched = native["scenarios"][native_key]["methods"][method]["pearson"]
                ax_native.plot([index - 0.08, index], [constant, matched], color=color, lw=0.6, ls=":")
                ax_native.scatter(
                    index - 0.08,
                    constant,
                    facecolor="white",
                    edgecolor=color,
                    marker=marker,
                    s=17,
                    linewidth=0.85,
                    zorder=5,
                )
    widths = [native["scenarios"][native_key]["native_window_bp"]["median"] / 1e6 for _short, _key, native_key in SCENARIOS]
    ax_native.set_xticks(x, [short for short, _key, _native_key in SCENARIOS])
    ax_native.set_xlim(-0.25, len(SCENARIOS) - 0.02)
    ax_native.set_ylim(0.70, 1.005)
    ax_native.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax_native.set_ylabel("Pearson $r$")
    ax_native.grid(axis="y", color="0.90", lw=0.5)
    ax_native.set_title("ReLERNN intended regime", loc="left", pad=3)
    ax_native.text(
        0.01,
        0.04,
        "native-window medians: " + ", ".join(f"{width:.2f}" for width in widths) + " Mb",
        transform=ax_native.transAxes,
        fontsize=4.2,
        color="0.38",
        va="bottom",
    )

    # b | Held-out interval calibration.
    ax = fig.add_subplot(top[0, 1])
    nominal = np.asarray(summary["heldout"]["coverage_curve"]["nominal"])
    empirical = np.asarray(summary["heldout"]["coverage_curve"]["empirical"])
    ax.plot([0.45, 1.0], [0.45, 1.0], color="0.6", ls="--", lw=1)
    ax.plot(nominal, empirical, color=BLUE, marker="o", ms=4, lw=1.4)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Calibration of intervals", loc="left")
    panel_label(ax, "b")

    # c | Fine-scale rate calibration remains restricted to comparable methods.
    ax = fig.add_subplot(top[0, 2])
    offsets = (-0.12, 0.12)
    for offset, (method, label, color, marker) in zip(offsets, METHODS[:2]):
        values = [fine_scale_record(summary, demography, key, method)["bias_ratio"] for _short, key, _native_key in SCENARIOS]
        ax.scatter(x + offset, values, color=color, marker=marker, s=18, zorder=3, label=label)
        if method == "pyrho":
            for index, (_short, key, _native_key) in enumerate(SCENARIOS[1:3], start=1):
                constant = paired_record(demography, key, method, "constant")["bias_ratio"]
                matched = paired_record(demography, key, method, "matched")["bias_ratio"]
                position = index + offset
                ax.plot([position - 0.07, position], [constant, matched], color=color, lw=0.6, ls=":")
                ax.scatter(position - 0.07, constant, facecolor="white", edgecolor=color, marker=marker, s=18, linewidth=0.8, zorder=4)
    ax.axhline(1, color="0.25", lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 7.2)
    ax.set_yticks([0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(x, [short for short, _key, _native_key in SCENARIOS], rotation=42, ha="right")
    ax.set_ylabel("median estimated / true rate")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    ax.set_title("Calibration of rates", loc="left")
    panel_label(ax, "c")

    # d | Two measured endpoints make stage inclusion explicit.
    ax = fig.add_subplot(bottom[0, 0])
    timing = {
        "fastrho": (12.5, 12.5),
        "pyrho": (176.7, 187.6),
        "relernn": (29.9, 10064.8),
    }
    for method, label, color, marker in METHODS:
        values = timing[method]
        ax.plot([0, 1], values, color=color, marker=marker, ms=4.0, lw=1.0, label=label)
    ax.set_yscale("log")
    ax.set_xlim(-0.08, 1.05)
    ax.set_ylim(8, 2.0e4)
    ax.set_xticks([0, 1], ["predict / infer\nonly", "full per-dataset\nworkflow"])
    ax.set_yticks([10, 60, 600, 3600, 10080], ["10 s", "1 min", "10 min", "1 h", "2.8 h"])
    ax.set_ylabel("Measured wall-clock")
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.legend(frameon=False, fontsize=5.3, loc="upper left")
    ax.text(0.55, 210, "+ lookup table", fontsize=4.7, color=BLACK)
    ax.text(0.55, 1800, "+ simulation\n+ training", fontsize=4.7, color=GRAY)
    ax.text(0.27, 15.5, "training amortized across datasets", fontsize=4.7, color=BLUE)
    ax.set_title("24 × 2-Mb bottleneck data", loc="left")
    panel_label(ax, "d")

    # e | Existing independent SLiM stress test.
    linked = bottom[0, 1].subgridspec(1, 2, wspace=0.42)
    ax_shape = fig.add_subplot(linked[0, 0])
    records = {record["name"]: record for record in selection["conditions"]}
    shape_conditions = [("neutral", records["neutral"]), ("BGS", records["bgsint_4"]), ("hard\nsweep", records["compl_7"])]
    xx = np.arange(len(shape_conditions))
    for method, color, marker in (("fastrho", BLUE, "o"), ("pyrho", BLACK, "s")):
        key = "fastrho_cmn_25kb" if method == "fastrho" else "pyrho_25kb"
        ax_shape.plot(xx, [record[key][0] for _label, record in shape_conditions], color=color, marker=marker, lw=1.1, ms=4, label=method)
    ax_shape.set_xticks(xx, [label for label, _record in shape_conditions])
    ax_shape.set_ylim(0.68, 0.93)
    ax_shape.set_ylabel("Pearson $r$ at 25 kb")
    ax_shape.legend(frameon=False, fontsize=5.5, loc="lower left")
    ax_shape.set_title("SLiM: map shape", loc="left")
    panel_label(ax_shape, "e")

    ax_scale = fig.add_subplot(linked[0, 1])
    xx = np.arange(2)
    for method, color, marker in (("fastrho", BLUE, "o"), ("pyrho", BLACK, "s")):
        values = []
        for condition in ("neutral", "sweep"):
            truth = np.asarray(selection_windows[f"calib_true_{condition}"], dtype=float)
            predicted = np.asarray(selection_windows[f"calib_{method}_{condition}"], dtype=float)
            keep = np.isfinite(truth) & np.isfinite(predicted) & (truth > 0) & (predicted > 0)
            values.append(float(np.median(predicted[keep] / truth[keep])))
        ax_scale.plot(xx, values, color=color, marker=marker, lw=1.1, ms=4)
    ax_scale.axhline(1, color="0.4", lw=0.8, ls=":")
    ax_scale.set_xticks(xx, ["neutral", "hard\nsweep"])
    ax_scale.set_ylim(0.45, 1.03)
    ax_scale.set_ylabel("median estimated / true")
    ax_scale.text(0.97, 0.87, "1 = correct scale", transform=ax_scale.transAxes, ha="right", va="top", fontsize=5.3, color="0.35")
    ax_scale.set_title("SLiM: rate scale", loc="left")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-json", required=True, type=Path)
    parser.add_argument("--misspecified-json", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    style()
    summary = json.loads((ROOT / "paper/results_snapshot/summary.json").read_text())
    demography = json.loads((ROOT / "paper/results_snapshot/demography_matched.json").read_text())
    selection = json.loads((ROOT / "paper/figdata/selection_dr.json").read_text())
    selection_windows = np.load(ROOT / "paper/figdata/selection_dr_figdata.npz")
    render(
        summary,
        demography,
        json.loads(args.native_json.read_text()),
        json.loads(args.misspecified_json.read_text()),
        selection,
        selection_windows,
        args.output,
    )


if __name__ == "__main__":
    main()
