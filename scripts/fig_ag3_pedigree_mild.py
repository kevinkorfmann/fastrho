"""Render the Ag3 pedigree comparison in the current Phase 2 three-panel grammar.

This deliberately keeps the manuscript's workflow/scatter/spatial-null layout while
replacing the underlying Phase 2 cross result with the audited Ag3 held-out analysis.
The all-15 and 2-Mb sensitivity results remain in the source table for the SI.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr

import paper_style as ps


AUTOSOMES = ("2R", "2L", "3R", "3L")
ARM_MARKERS = {"2R": "o", "2L": "s", "3R": "^", "3L": "D"}
BLUE = ps.C["fastrho"]
INK = ps.C["truth"]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def box(ax, xy, width, height, label, edge, face="white", fontsize=7.0):
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=face,
            transform=ax.transAxes,
        )
    )
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.15,
    )


def spatial_null(rows: list[dict[str, str]]) -> tuple[float, np.ndarray]:
    by_arm = {
        arm: sorted(
            (row for row in rows if row["arm"] == arm),
            key=lambda row: int(row["start_bp"]),
        )
        for arm in AUTOSOMES
    }
    direct = [
        float(row["direct_arm_normalized"])
        for arm in AUTOSOMES
        for row in by_arm[arm]
        if row["direct_arm_normalized"]
    ]
    atlas = [
        float(row["atlas_arm_normalized"])
        for arm in AUTOSOMES
        for row in by_arm[arm]
        if row["direct_arm_normalized"]
    ]
    observed = float(spearmanr(direct, atlas).statistic)
    null = []
    for shifts in itertools.product(*(range(len(by_arm[arm])) for arm in AUTOSOMES)):
        if not any(shifts):
            continue
        shifted = []
        for arm, shift in zip(AUTOSOMES, shifts, strict=True):
            arm_rows = by_arm[arm]
            values = np.asarray(
                [float(row["atlas_arm_normalized"]) for row in arm_rows], float
            )
            supported = np.asarray(
                [bool(row["direct_arm_normalized"]) for row in arm_rows], bool
            )
            shifted.extend(np.roll(values, shift)[supported])
        null.append(float(spearmanr(direct, shifted).statistic))
    return observed, np.asarray(null)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--windows",
        type=Path,
        default=Path(
            "legacy/pre-phase2-snapshot/paper/tables/ag3_pedigree_windows.tsv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "legacy/pre-phase2-snapshot/paper/tables/ag3_pedigree_summary.tsv"
        ),
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path(
            "legacy/pre-phase2-snapshot/paper/results_snapshot/ag3_pedigree.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/figures/fig_ag3_pedigree_mild.pdf"),
    )
    args = parser.parse_args()

    rows = [
        row
        for row in read_tsv(args.windows)
        if row["analysis"] == "heldout" and row["window_bp"] == "5000000"
    ]
    supported = [row for row in rows if row["direct_arm_normalized"]]
    primary = next(
        row
        for row in read_tsv(args.summary)
        if row["analysis"] == "heldout" and row["window_bp"] == "5000000"
    )
    result = json.loads(args.result.read_text())
    observed, null = spatial_null(rows)

    ps.style()
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
    ax_a.set_title("Ag3 pedigree comparison setup", loc="left", pad=5)
    box(
        ax_a,
        (0.03, 0.69),
        0.41,
        0.23,
        "Ag3 pedigrees\n5 held-out families\n97 crossovers",
        BLUE,
        "#F3F4FF",
    )
    box(
        ax_a,
        (0.56, 0.69),
        0.41,
        0.23,
        "Inferred maps\n10 populations",
        INK,
        "#F5F5F3",
    )
    box(
        ax_a,
        (0.12, 0.40),
        0.76,
        0.17,
        "Common 5-Mb windows\nwithin-arm normalization",
        INK,
    )
    box(
        ax_a,
        (0.12, 0.14),
        0.76,
        0.17,
        "Spearman $r_s$\nwithin-arm shift test",
        INK,
    )
    for start, end in (
        ((0.235, 0.69), (0.34, 0.57)),
        ((0.765, 0.69), (0.66, 0.57)),
        ((0.50, 0.40), (0.50, 0.31)),
    ):
        ax_a.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax_a.transAxes,
            textcoords=ax_a.transAxes,
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#888888"},
        )
    qc = result["caller_simulation_heldout"]
    ax_a.text(
        0.5,
        -0.03,
        f"Caller checks: {100 * qc['count_bias_fraction']:.2f}% bias; "
        f"{100 * qc['truth_interval_coverage']:.1f}% coverage;\n"
        f"{100 * qc['observed_background_spike_in_recovery_fraction']:.1f}% spike-in recovery",
        transform=ax_a.transAxes,
        ha="center",
        va="top",
        fontsize=6.3,
        color="#666666",
    )

    for arm, marker in ARM_MARKERS.items():
        arm_rows = [row for row in supported if row["arm"] == arm]
        ax_b.scatter(
            [float(row["atlas_arm_normalized"]) for row in arm_rows],
            [float(row["direct_arm_normalized"]) for row in arm_rows],
            s=28,
            marker=marker,
            facecolor=BLUE,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.85,
            zorder=3,
        )
    limit = 1.08 * max(
        max(float(row["atlas_arm_normalized"]) for row in supported),
        max(float(row["direct_arm_normalized"]) for row in supported),
    )
    ax_b.set(xlim=(-0.08, limit), ylim=(-0.08, limit))
    ax_b.set_xlabel("Atlas rate (within-arm relative)")
    ax_b.set_ylabel("Pedigree rate (within-arm relative)")
    ax_b.set_title("Consistent broad-scale spatial ordering", loc="left", pad=5)
    ax_b.text(
        0.04,
        0.96,
        rf"$r_s={observed:.2f}$"
        "\n"
        rf"$P={float(primary['circular_shift_p_two_sided']):.3f}$"
        "\n"
        rf"{len(supported)} windows",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        linespacing=1.2,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker=marker,
            ls="",
            markerfacecolor=BLUE,
            markeredgecolor="white",
            label=arm,
        )
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

    ax_c.hist(null, bins=24, color="#C8C8C8", edgecolor="white", linewidth=0.35)
    ax_c.axvline(observed, color=BLUE, lw=1.8)
    ax_c.axvline(-observed, color=BLUE, lw=1.0, ls=(0, (3, 2)))
    ax_c.set_xlim(-0.8, 0.8)
    ax_c.text(
        0.82,
        0.96,
        f"observed $r_s={observed:.2f}$\n"
        f"{len(null):,} spatial shifts\n"
        f"two-sided $P={float(primary['circular_shift_p_two_sided']):.4f}$",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )
    ax_c.set_xlabel(r"Shifted-map Spearman $r_s$")
    ax_c.set_ylabel("Circular shifts")
    ax_c.set_title("Within-arm spatial-shift test", loc="left", pad=5)

    for label, axis in zip("abc", (ax_a, ax_b, ax_c), strict=True):
        ps.panel(axis, label, x=-0.14, y=1.10, fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
