"""Render the fixed Arabis inferred recombination versus F2 cross benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import paper_style as ps

ROOT = Path(__file__).resolve().parents[1]
ARRAYS = ROOT / "paper" / "figdata" / "arabis_cross.npz"
STRUCTURED_ARRAYS = ROOT / "paper" / "figdata" / "arabis_cross_structured.npz"
SUMMARY = ROOT / "paper" / "results_snapshot" / "arabis_cross.json"
SMALLN_SUMMARY = ROOT / "paper" / "results_snapshot" / "arabis_cross_smalln.json"
STRUCTURED_SUMMARY = ROOT / "paper" / "results_snapshot" / "arabis_cross_structured.json"
WINDOW_DIAGNOSTICS = (
    ROOT / "paper" / "results_snapshot" / "arabis_window_diagnostics.json"
)
COLORS = {"nemorensis": ps.CB[6], "sagittata": ps.CB[5], "consensus": ps.C["fastrho"]}
MARKERS = {"nemorensis": "o", "sagittata": "s", "consensus": "D"}
LABELS = {
    "nemorensis": r"$A.\ nemorensis$",
    "sagittata": r"$A.\ sagittata$",
    "consensus": "species consensus",
}


def diagnostic_rows(diagnostics: dict, species: str, chrom: str) -> list[dict]:
    return sorted(
        (
            row
            for row in diagnostics["windows"]
            if row["species"] == species and row["chromosome"] == chrom
        ),
        key=lambda row: row["window_index"],
    )


def cumulative_map(relative_rates: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Integrate window rates and normalize each map to end at one."""
    rates = np.atleast_2d(np.asarray(relative_rates, float))
    increments = rates * np.asarray(widths, float)[None, :]
    cumulative = np.column_stack(
        (np.zeros(rates.shape[0]), np.cumsum(increments, axis=1))
    )
    cumulative /= cumulative[:, -1, None]
    return cumulative[0] if np.asarray(relative_rates).ndim == 1 else cumulative


def consensus_seed_cumulative_maps(
    diagnostics: dict, chrom: str, widths: np.ndarray
) -> np.ndarray:
    nem = diagnostic_rows(diagnostics, "nemorensis", chrom)
    sag = diagnostic_rows(diagnostics, "sagittata", chrom)
    nem_seeds = np.asarray([row["seed_relative_rates"] for row in nem], float).T
    sag_seeds = np.asarray([row["seed_relative_rates"] for row in sag], float).T
    consensus = np.sqrt(nem_seeds * sag_seeds)
    return cumulative_map(consensus, widths)


def main() -> None:
    ps.style()
    arrays = np.load(ARRAYS, allow_pickle=False)
    structured_arrays = np.load(STRUCTURED_ARRAYS, allow_pickle=False)
    summary = json.loads(SUMMARY.read_text())
    smalln = json.loads(SMALLN_SUMMARY.read_text())
    structured = json.loads(STRUCTURED_SUMMARY.read_text())
    diagnostics = json.loads(WINDOW_DIAGNOSTICS.read_text())
    fig = plt.figure(figsize=(7.15, 6.55))
    outer = fig.add_gridspec(
        3, 1, height_ratios=(1.12, 0.92, 0.72), hspace=0.72
    )

    summaries = outer[0].subgridspec(
        1, 3, width_ratios=(1.16, 0.78, 1.06), wspace=0.48
    )
    cross = np.concatenate([structured_arrays[f"chr{i}_cross"] for i in range(1, 9)])

    ax_scatter = fig.add_subplot(summaries[0])
    scatter_values = [cross]
    for species in ("nemorensis", "sagittata", "consensus"):
        pred = np.concatenate(
            [structured_arrays[f"chr{i}_{species}"] for i in range(1, 9)]
        )
        scatter_values.append(pred)
        rho = structured["resolutions"]["2Mb"]["maps"][species]["spearman"]
        is_consensus = species == "consensus"
        ax_scatter.scatter(
            cross,
            pred,
            s=17 if is_consensus else 10,
            alpha=0.58 if is_consensus else 0.30,
            color=COLORS[species],
            marker=MARKERS[species],
            linewidth=0,
            zorder=3 if is_consensus else 2,
            label=f"{LABELS[species]}, $r_s={rho:.2f}$",
        )
    limit = float(np.max(np.concatenate(scatter_values)) * 1.04)
    ax_scatter.plot([0, limit], [0, limit], color="#888888", lw=0.75, ls=":")
    ax_scatter.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="F2-map relative rate",
        ylabel="population-LD relative rate",
    )
    ax_scatter.legend(loc="upper left", fontsize=5.9, handletextpad=0.3)
    ax_scatter.set_title(
        r"$\bf{a}$  Local 2-Mb rate agreement",
        loc="left",
        fontsize=8.7,
    )

    ax_topology = fig.add_subplot(summaries[1])
    topology = structured["resolutions"]["2Mb"][
        "outer_quarters_to_center_half_ratio"
    ]
    species_order = ("nemorensis", "sagittata", "consensus")
    topo_x = np.arange(3, dtype=float)
    ax_topology.axhline(1, color="#999999", lw=0.65, ls=":")
    ax_topology.axhline(
        topology["cross"], color=ps.C["truth"], lw=1.15, label=f"F2 = {topology['cross']:.2f}"
    )
    for xpos, species in zip(topo_x, species_order):
        value = topology[species]
        ax_topology.scatter(
            xpos,
            value,
            s=40,
            color=COLORS[species],
            marker=MARKERS[species],
            zorder=3,
        )
        ax_topology.text(
            xpos,
            value + 0.07,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            color=COLORS[species],
        )
    ax_topology.set_xticks(topo_x, ["nem.", "sag.", "cons."])
    ax_topology.tick_params(axis="x", labelsize=6.5)
    ax_topology.set_xlim(-0.45, 2.45)
    ax_topology.set_ylim(0.85, 2.24)
    ax_topology.set_ylabel("outer quarters / centre")
    ax_topology.legend(loc="lower right", fontsize=6.2)
    ax_topology.set_title(
        r"$\bf{b}$  Arm-to-centre pattern", loc="left", fontsize=8.7
    )

    ax_sens = fig.add_subplot(summaries[2])
    widths = np.asarray([1, 2, 5], float)
    offsets = {"nemorensis": -0.11, "sagittata": 0.0, "consensus": 0.11}
    for species in ("nemorensis", "sagittata", "consensus"):
        rows = [
            structured["resolutions"][f"{w}Mb"]["maps"][species]
            for w in (1, 2, 5)
        ]
        y = np.asarray([row["spearman"] for row in rows])
        ci = np.asarray([row["spearman_chromosome_bootstrap_ci95"] for row in rows])
        xpos = widths + offsets[species]
        ax_sens.vlines(xpos, ci[:, 0], ci[:, 1], color=COLORS[species], lw=1.35)
        ax_sens.plot(
            xpos,
            y,
            color=COLORS[species],
            marker=MARKERS[species],
            lw=1.35,
            ms=5,
            label=LABELS[species],
        )
        robust = rows[1]["spearman_excluding_distorted_chr4_chr7"]
        ax_sens.scatter(
            2 + offsets[species],
            robust,
            s=45,
            facecolor="white",
            edgecolor=COLORS[species],
            marker=MARKERS[species],
            lw=1.1,
            zorder=4,
        )
    ax_sens.axhline(0, color="#777777", lw=0.75, ls=":")
    ax_sens.set_ylim(-0.30, 1.0)
    ax_sens.set_xticks(widths)
    ax_sens.set_xlabel("window size (Mb)")
    ax_sens.set_ylabel("Spearman $r_s$")
    ax_sens.set_title(
        r"$\bf{c}$  Agreement across 1-5 Mb",
        loc="left",
        fontsize=8.7,
    )
    ax_sens.text(
        0.98,
        0.05,
        "open: exclude chr 4, 7",
        transform=ax_sens.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#555555",
    )

    detail_grid = outer[1].subgridspec(
        1, 2, width_ratios=(1.55, 0.85), wspace=0.40
    )
    ax_pooled = fig.add_subplot(detail_grid[0])
    grid = np.linspace(0, 1, 201)
    cross_curves = []
    structured_curves = []
    seed_curves = []
    for i in range(1, 9):
        chrom = f"chr{i}"
        edges = arrays[f"{chrom}_edges"] / 1e6
        physical_fraction = (edges - edges[0]) / (edges[-1] - edges[0])
        bin_widths = np.diff(edges)
        cross_curves.append(
            np.interp(
                grid,
                physical_fraction,
                cumulative_map(arrays[f"{chrom}_cross"], bin_widths),
            )
        )
        structured_curves.append(
            np.interp(
                grid,
                physical_fraction,
                cumulative_map(
                    structured_arrays[f"{chrom}_consensus"], bin_widths
                ),
            )
        )
        seed_curves.append(
            np.asarray(
                [
                    np.interp(grid, physical_fraction, curve)
                    for curve in consensus_seed_cumulative_maps(
                        diagnostics, chrom, bin_widths
                    )
                ]
            )
        )
    cross_mean = np.mean(cross_curves, axis=0)
    structured_mean = np.mean(structured_curves, axis=0)
    seed_means = np.mean(seed_curves, axis=0)
    ax_pooled.axvline(25, color="#A8A8A4", lw=0.65, ls=(0, (3, 2)), zorder=0)
    ax_pooled.axvline(75, color="#A8A8A4", lw=0.65, ls=(0, (3, 2)), zorder=0)
    ax_pooled.plot((0, 100), (0, 100), color="#999999", lw=0.65, ls=":", zorder=1)
    seed_lo = np.min(seed_means, axis=0) * 100
    seed_hi = np.max(seed_means, axis=0) * 100
    ax_pooled.plot(
        grid * 100,
        seed_lo,
        color=COLORS["consensus"],
        alpha=0.55,
        linewidth=0.65,
        linestyle=(0, (2, 2)),
        zorder=2,
    )
    ax_pooled.plot(
        grid * 100,
        seed_hi,
        color=COLORS["consensus"],
        alpha=0.55,
        linewidth=0.65,
        linestyle=(0, (2, 2)),
        zorder=2,
    )
    ax_pooled.plot(
        grid * 100,
        cross_mean * 100,
        color=ps.C["truth"],
        lw=1.55,
        zorder=3,
    )
    ax_pooled.plot(
        grid * 100,
        structured_mean * 100,
        color=COLORS["consensus"],
        lw=1.55,
        zorder=4,
    )
    pooled_handles = (
        Line2D([], [], color=ps.C["truth"], lw=1.25, label="F2 cross map"),
        Line2D([], [], color=COLORS["consensus"], lw=1.25, label="structured consensus"),
        Line2D([], [], color=COLORS["consensus"], lw=0.7, ls=(0, (2, 2)),
               label="seven-model bounds"),
    )
    ax_pooled.legend(
        handles=pooled_handles,
        loc="upper left",
        ncol=1,
        fontsize=5.8,
        handlelength=1.4,
        labelspacing=0.25,
    )
    ax_pooled.text(
        0.99,
        0.05,
        "all 8 chromosomes; equal weight",
        transform=ax_pooled.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.3,
        color="#555555",
    )
    ax_pooled.set(
        xlim=(0, 100),
        ylim=(0, 100),
        xlabel="physical position (%)",
        ylabel="cumulative genetic position (%)",
    )
    ax_pooled.set_xticks((0, 25, 50, 75, 100))
    ax_pooled.set_yticks((0, 25, 50, 75, 100))
    ax_pooled.set_title(
        r"$\bf{d}$  Pooled chromosome topology",
        loc="left",
        fontsize=8.7,
    )

    ax_local = fig.add_subplot(detail_grid[1])
    local_pred = np.concatenate(
        [structured_arrays[f"chr{i}_consensus"] for i in range(1, 9)]
    )
    quantile_edges = np.quantile(cross, np.linspace(0, 1, 6))
    quantile_index = np.searchsorted(
        quantile_edges[1:-1], cross, side="right"
    )
    groups = [local_pred[quantile_index == i] for i in range(5)]
    rng = np.random.default_rng(109)
    for i, values in enumerate(groups, start=1):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax_local.scatter(
            i + jitter,
            values,
            s=10,
            color=COLORS["consensus"],
            alpha=0.28,
            linewidth=0,
            zorder=2,
        )
    boxes = ax_local.boxplot(
        groups,
        positions=np.arange(1, 6),
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["consensus"], "lw": 1.35},
        boxprops={"facecolor": "white", "edgecolor": COLORS["consensus"], "lw": 0.9},
        whiskerprops={"color": COLORS["consensus"], "lw": 0.8},
        capprops={"color": COLORS["consensus"], "lw": 0.8},
    )
    for box in boxes["boxes"]:
        box.set_alpha(0.92)
    top_bottom = float(np.median(groups[-1]) / np.median(groups[0]))
    ax_local.axhline(1, color="#888888", lw=0.65, ls=":", zorder=1)
    ax_local.text(
        0.97,
        0.95,
        f"top / bottom median = {top_bottom:.2f}x",
        transform=ax_local.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color="#444444",
    )
    ax_local.text(
        0.97,
        0.84,
        "20-21 windows per quintile",
        transform=ax_local.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color="#555555",
    )
    ax_local.set(
        xlim=(0.55, 5.45),
        ylim=(0, 2.8),
        xlabel="F2-map rate quintile",
        ylabel="structured relative rate",
    )
    ax_local.set_xticks((1, 2, 3, 4, 5), ("low", "2", "3", "4", "high"))
    ax_local.set_yticks((0, 1, 2))
    ax_local.set_title(
        r"$\bf{e}$  Local 2-Mb high-rate enrichment",
        loc="left",
        fontsize=8.7,
    )

    ax_smalln = fig.add_subplot(outer[2])
    x = np.arange(len(species_order), dtype=float)
    for xpos, species in zip(x, species_order):
        baseline = summary["resolutions"]["2Mb"]["maps"][species]
        matched = smalln["resolutions"]["2Mb"]["maps"][species]
        structure_aware = structured["resolutions"]["2Mb"]["maps"][species]
        ax_smalln.plot(
            [xpos - 0.14, xpos, xpos + 0.14],
            [baseline["spearman"], matched["spearman"], structure_aware["spearman"]],
            color="#999999",
            lw=0.85,
            zorder=1,
        )
        families = (
            (-0.14, baseline, "white"),
            (0.0, matched, "#B8B8B8"),
            (0.14, structure_aware, COLORS[species]),
        )
        for offset, row, facecolor in families:
            lo, hi = row["spearman_chromosome_bootstrap_ci95"]
            ax_smalln.vlines(xpos + offset, lo, hi, color=COLORS[species], lw=1.35)
            ax_smalln.scatter(
                xpos + offset,
                row["spearman"],
                s=36,
                marker=MARKERS[species],
                facecolor=facecolor,
                edgecolor=COLORS[species],
                lw=1.1,
                zorder=3,
            )
    ax_smalln.scatter([], [], s=34, facecolor="white", edgecolor="#333333",
                      label="initial model")
    ax_smalln.scatter([], [], s=34, facecolor="#B8B8B8", edgecolor="#333333",
                      label="sample-size matched")
    ax_smalln.scatter([], [], s=34, facecolor="#333333", edgecolor="#333333",
                      label="structured selfing")
    ax_smalln.axhline(0, color="#777777", lw=0.75, ls=":")
    ax_smalln.set_xticks(x, [LABELS[species] for species in species_order])
    ax_smalln.set_xlim(-0.45, 2.45)
    ax_smalln.set_ylabel("$r_s$ with F2 map")
    ax_smalln.set_title(
        r"$\bf{f}$  Model-family comparison at 2 Mb",
        loc="left",
        fontsize=8.7,
    )
    ax_smalln.legend(loc="upper left", ncol=3, fontsize=6.8)

    ps.save(fig, "fig_si_arabis_cross", formats=("pdf", "png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
