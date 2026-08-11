"""Render SI-only panels after removing material duplicated in the main figures.

Each output retains the unique evidence from a formerly mixed supplementary figure.
Source arrays and the statistical estimands are unchanged; this script changes only
panel selection and layout so the SI does not repeat main-text graphics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fig_calibration as calibration
import fig_identifiability as identifiability
import fig_relernn_scale as relernn
import fig_selfer_ceiling as selfer
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import paper_style as ps

ROOT = Path(__file__).resolve().parents[1]
FIGDATA = ROOT / "paper" / "figdata"
RESULTS = ROOT / "paper" / "results_snapshot"
OUT = ROOT / "paper" / "figures"


def _relabel(
    ax, old: str, new: str, x: float = -0.13, y: float = 1.06, fontsize: float = 14
) -> None:
    """Replace an imported panel's letter and leading title marker."""
    ax.set_title(re.sub(rf"^\({old}\)", f"({new})", ax.get_title()), loc="left")
    for artist in list(ax.texts):
        if artist.get_text() == old:
            artist.remove()
    ps.panel(ax, new, x=x, y=y, fontsize=fontsize)


def _finish(fig, stem: str, **tight_kwargs) -> None:
    fig.tight_layout(**tight_kwargs)
    ps.save(fig, stem, outdir=str(OUT), formats=("pdf", "png"), dpi=600)
    plt.close(fig)


def _draw_canid_severity(ax) -> None:
    d = np.load(FIGDATA / "dog_fig.npz", allow_pickle=True)
    ne, own, transfer = d["Ne"], d["own"], d["trn"]
    bx, b_own, b_transfer = d["d_bx"], d["d_own"], d["d_trn"]
    ax.scatter(ne, own, s=13, color=ps.TRANSFER["breed"], alpha=0.22, lw=0)
    ax.scatter(ne, transfer, s=13, color=ps.TRANSFER["village"], alpha=0.22, lw=0)
    ax.vlines(bx, b_own, b_transfer, color="#D0D0CC", linewidth=0.65, zorder=0)
    ax.plot(bx, b_transfer, "o-", color=ps.TRANSFER["village"], label="large-population transfer")
    ax.plot(bx, b_own, "s-", color=ps.TRANSFER["breed"], label="breed, own data")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.xaxis.set_major_locator(mticker.FixedLocator([50, 100, 200, 500]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["50", "100", "200", "500"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlabel(r"Present breed $N_e$  (deeper bottleneck $\longrightarrow$)")
    ax.set_ylabel("Map recovery (100-kb log-Pearson $r$)")
    ax.legend(loc="lower center")


def canid_severity() -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.55))
    _draw_canid_severity(ax)
    ps.panel(ax, "a", x=-0.10)
    _finish(fig, "fig_si_canid_severity")


def selfer_confounds() -> None:
    d = json.loads((FIGDATA / "selfer_ceiling.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.8, 3.55))
    selfer.panel_decomp(ax, d)
    _relabel(ax, "c", "a", x=-0.18)
    ax.set_yticklabels([label.get_text().replace("→", " to ") for label in ax.get_yticklabels()])
    ax.set_title("")
    _finish(fig, "fig_si_selfer_confounds")


def demographic_mating_limits() -> None:
    """Combine the canid and selfing limits into one SI figure."""
    d = json.loads((FIGDATA / "selfer_ceiling.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 4.75))
    _draw_canid_severity(axes[0])
    axes[0].set_ylabel("Map recovery\n(100-kb log-Pearson $r$)")
    for line in axes[0].lines:
        line.set_linewidth(min(line.get_linewidth(), 1.6))
        line.set_markersize(min(line.get_markersize(), 4.5))
    for collection in axes[0].collections:
        if hasattr(collection, "get_sizes"):
            sizes = collection.get_sizes()
            if len(sizes):
                collection.set_sizes(np.minimum(sizes, 9))
    axes[0].legend(
        loc="lower center", fontsize=7.1, handlelength=1.5, labelspacing=0.25,
        borderaxespad=0.35,
    )
    ps.panel(axes[0], "a", x=-0.07, y=1.04, fontsize=10)
    selfer.panel_decomp(axes[1], d)
    _relabel(axes[1], "c", "b", x=-0.11, y=1.04, fontsize=10)
    axes[1].set_yticklabels(
        [label.get_text().replace("→", " to ") for label in axes[1].get_yticklabels()]
    )
    axes[1].set_title("")
    for ax in axes:
        ax.tick_params(axis="both", labelsize=7.2)
        ax.xaxis.label.set_size(8.2)
        ax.yaxis.label.set_size(8.2)
        for text in ax.texts:
            if text.get_text() not in {"a", "b"}:
                text.set_fontsize(min(text.get_fontsize(), 7.3))
    _finish(fig, "fig_si_demography_mating_limits", h_pad=1.15)


def regime_probes() -> None:
    ident = json.loads((FIGDATA / "identifiability.json").read_text(encoding="utf-8"))
    mechanism = json.loads((FIGDATA / "mechanism.json").read_text(encoding="utf-8"))
    fig = plt.figure(figsize=(7.15, 5.6))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.15, 0.85), hspace=0.52)
    ax_ident = fig.add_subplot(grid[0])
    ax_mechanism = fig.add_subplot(grid[1])
    identifiability.panel_identifiability(ax_ident, ident)
    identifiability.panel_mechanism(ax_mechanism, mechanism)
    _relabel(ax_ident, "a", "a", x=-0.07)
    _relabel(ax_mechanism, "b", "b", x=-0.07)
    fig.savefig(OUT / "fig_si_regime_probes.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_si_regime_probes.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def resolution_detail() -> None:
    data = np.load(FIGDATA / "relernn_showdown.npz", allow_pickle=True)
    meta = json.loads(str(data["meta"]))
    grid_kb = np.asarray(meta["grids_kb"], dtype=float)
    colors = {"fastrho": ps.C["fastrho"], "pyrho": ps.C["pyrho"], "relernn": ps.C["relernn"]}
    labels = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}
    maps = np.load(FIGDATA / "realmaps.npz", allow_pickle=True)
    fig = plt.figure(figsize=(7.15, 7.35))
    layout = fig.add_gridspec(3, 1, height_ratios=(0.9, 1.0, 1.15), hspace=0.54)
    top = fig.add_subplot(layout[0])
    bottom = fig.add_subplot(layout[1])
    top.axvline(100, color="#B0B0AC", linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    method_style = {
        "fastrho": ("o", "-"),
        "pyrho": ("s", (0, (4, 2.5))),
        "relernn": ("^", (0, (1.5, 2))),
    }
    for method in ("fastrho", "pyrho", "relernn"):
        marker, linestyle = method_style[method]
        top.plot(
            grid_kb,
            meta["curve"][method],
            color=colors[method],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.5,
            markersize=4.2,
            label=labels[method],
        )
    top.set_xscale("log")
    top.invert_xaxis()
    top.set_ylim(0, 1)
    top.set_xlabel(r"Scoring window (kb): coarse $\leftarrow$ $\rightarrow$ fine")
    top.set_ylabel("Pearson $r$ with true map")
    top.legend(loc="lower left", ncol=3)
    ps.panel(top, "a", x=-0.08)

    repro = np.load(FIGDATA / "repro_showdown.npz", allow_pickle=True)
    x = np.asarray(repro["centers_hi_mb"], dtype=float)
    floor = 1.2e-9
    truth = np.clip(np.asarray(repro["truth_hi"], dtype=float), floor, None)
    series = (
        ("ReLERNN", ps.C["relernn"], np.asarray(repro["relernn_hi"])),
        ("true map", ps.C["truth"], truth),
        ("pyrho", ps.C["pyrho"], np.asarray(repro["pyrho_hi"])),
        ("fastrho", ps.C["fastrho"], np.asarray(repro["fastrho_hi"])),
    )
    for label, color, values in series:
        linestyle = (0, (4, 2.5)) if label == "true map" else "-"
        bottom.plot(
            x,
            np.clip(values, floor, None),
            color=color,
            linewidth=1.5,
            linestyle=linestyle,
            label=label,
        )
    bottom.set_yscale("log")
    bottom.set_xlim(16.4, 19.2)
    bottom.set_ylim(1e-9, 1.6e-7)
    bottom.set_xlabel("Comeron chromosome-2L position (Mb)")
    bottom.set_ylabel("Rate (bp$^{-1}$)")
    bottom.legend(loc="upper center", ncol=4, fontsize=8.2)
    ps.panel(bottom, "b", x=-0.08)
    identifiability.panel_human(fig, layout[2], maps)
    human_top = fig.axes[-2]
    for human_ax in fig.axes[-2:]:
        for line in human_ax.lines:
            if line.get_linewidth() > 1.5:
                line.set_linewidth(1.5)
    _relabel(human_top, "d", "c", x=-0.08, y=1.10)
    fig.savefig(OUT / "fig_si_resolution.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_si_resolution.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def calibration_stress() -> None:
    paired = json.loads((RESULTS / "demography_matched.json").read_text(encoding="utf-8"))
    selection = np.load(FIGDATA / "selection_dr_figdata.npz", allow_pickle=True)
    # Author at the same oversized native scale as the dense input-view figure;
    # LaTeX down-scaling then gives comparable apparent fonts and line weights.
    fig = plt.figure(figsize=(11.0, 7.15))
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1.15), hspace=0.40, wspace=0.34)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    scenario_style = {
        "bottleneck": ("Bottleneck", ps.C["fastrho"]),
        "expansion": ("Expansion", ps.C["pyrho"]),
    }
    labels = ["ReLERNN", "pyrho"]
    x = np.arange(len(labels))
    offset = 0.09
    observed = {"pearson": [], "bias_ratio": []}
    for scenario, (label, color) in scenario_style.items():
        methods = paired["scenarios"][scenario]
        for axis, metric in zip(axes[:2], ("pearson", "bias_ratio")):
            for method_index, method in enumerate(("relernn", "pyrho")):
                constant = methods[method]["arms"]["constant"]["25kb"][metric]
                matched = methods[method]["arms"]["matched"]["25kb"][metric]
                observed[metric].extend((constant, matched))
                axis.plot(
                    [method_index - offset, method_index + offset],
                    [constant, matched],
                    color=color,
                    lw=1.2,
                    label=label if method_index == 0 else None,
                )
                axis.scatter(
                    method_index - offset,
                    constant,
                    facecolor="white",
                    edgecolor=color,
                    s=28,
                    zorder=3,
                )
                axis.scatter(method_index + offset, matched, color=color, s=28, zorder=3)
    axes[0].set_ylabel("Pearson $r$ at 25 kb")
    axes[0].set_ylim(
        min(-0.05, min(observed["pearson"]) - 0.08),
        min(1.0, max(0.75, max(observed["pearson"]) + 0.08)),
    )
    axes[0].set_title("Fine-scale map recovery", loc="left")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    axes[1].axhline(1, color="0.45", lw=0.8, ls="--")
    axes[1].set_yscale("log")
    axes[1].set_ylim(
        max(0.03, min(0.1, min(observed["bias_ratio"]) / 1.4)),
        max(3.2, max(observed["bias_ratio"]) * 1.35),
    )
    axes[1].set_ylabel("Median estimated / true rate")
    axes[1].set_title("Absolute scale", loc="left")
    for axis in axes[:2]:
        axis.set_xticks(x, labels)
        axis.set_xlim(-0.35, 1.35)
        axis.grid(axis="y", color="0.91", lw=0.5)
        axis.text(
            0.99,
            0.98,
            "open constant · filled matched",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=5.7,
            color="0.35",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.5},
        )
    calibration.panel_selection(axes[2], selection)
    axes[2].set_aspect("auto")
    _relabel(axes[2], "d", "c", x=-0.08, y=1.04)
    axes[2].set_title("Hard-sweep calibration", loc="left")
    ps.panel(axes[0], "a", x=-0.10, y=1.14)
    ps.panel(axes[1], "b", x=-0.13, y=1.14)
    _finish(fig, "fig_si_calibration_stress")


def relernn_mechanisms() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.25))
    relernn.panel_hotspot(axes[0])
    relernn.panel_steelman(axes[1])
    for ax in axes:
        for line in ax.lines:
            if line.get_linewidth() > 1.5:
                line.set_linewidth(1.5)
        for collection in ax.collections:
            collection.set_linewidths(
                [min(float(width), 1.5) for width in collection.get_linewidths()]
            )
    _relabel(axes[0], "b", "a", x=-0.12)
    _relabel(axes[1], "c", "b", x=-0.12)
    axes[0].set_title("Short-hotspot detectability", loc="left")
    axes[1].set_title("Matched per-SNP GRU baseline", loc="left")
    _finish(fig, "fig_si_relernn_mechanisms", w_pad=2.0)


def main() -> None:
    ps.style()
    canid_severity()
    selfer_confounds()
    demographic_mating_limits()
    regime_probes()
    resolution_detail()
    calibration_stress()
    relernn_mechanisms()


if __name__ == "__main__":
    main()
