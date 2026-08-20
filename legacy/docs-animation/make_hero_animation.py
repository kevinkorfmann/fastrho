#!/usr/bin/env python3
"""Build the documentation animation and static evaluation figure.

The hero uses a contiguous 10 Mb segment of a held-out msprime benchmark
simulated along the published Drosophila chromosome 2L Comeron landscape.
The static evaluation retains the separate deCODE-derived benchmark because
that archive also includes the model's conditional interval. Rendering is
deterministic and does not require a GPU.

Run from the repository root (dependencies are isolated from the inference environment):

    uv run --no-project --python 3.12 \
      --with-requirements docs/_scripts/requirements.txt \
      python docs/_scripts/make_hero_animation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
ANIMATION_SOURCE = ROOT / "paper" / "figdata" / "repro_showdown.npz"
EVALUATION_SOURCE = ROOT / "paper" / "figdata" / "relernn_showdown.npz"
OUTPUT = ROOT / "docs" / "_static" / "anim_inference.gif"
EVALUATION_OUTPUT = ROOT / "docs" / "_static" / "msprime_evaluation.png"

# Okabe-Ito blue plus neutral ink: legible in grayscale and color-vision deficiencies.
BLUE = "#0072B2"
SKY = "#8FD0EE"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#7B61A8"
INK = "#183247"
MUTED = "#66788A"
TRUTH = "#657786"
GRID = "#DCE5EB"
BACKGROUND = "#FBFDFE"

ANIMATION_LENGTH_MB = 10.0
EVALUATION_LENGTH_MB = 2.0
BIN_MB = 0.025
RATE_SCALE = 1e8  # per bp per generation -> cM/Mb


class PlayOncePillowWriter(PillowWriter):
    """Write one play with no GIF loop extension, then hold the final frame."""

    def finish(self) -> None:
        self._frames[0].save(
            self.outfile,
            save_all=True,
            append_images=self._frames[1:],
            duration=int(1000 / self.fps),
        )


def _bin_mean(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return the mean of ``y`` in every half-open bin of ``edges``."""
    index = np.digitize(x, edges) - 1
    result = np.full(edges.size - 1, np.nan)
    for i in range(result.size):
        selected = index == i
        if selected.any():
            result[i] = np.nanmean(y[selected])
    return result


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation for the finite values used in the displayed bins."""
    keep = np.isfinite(a) & np.isfinite(b)
    rank = lambda x: np.argsort(np.argsort(x))  # noqa: E731 - compact, deterministic ranks
    return float(np.corrcoef(rank(a[keep]), rank(b[keep]))[0, 1])


def load_evaluation_data() -> tuple[np.ndarray, ...]:
    """Load and place truth, prediction, and intervals on the same 25-kb grid."""
    archive = np.load(EVALUATION_SOURCE, allow_pickle=True)
    edges = np.arange(0, EVALUATION_LENGTH_MB + BIN_MB / 2, BIN_MB)
    centers = 0.5 * (edges[:-1] + edges[1:])

    map_positions = archive["mpos"] / 1e6
    map_centers = 0.5 * (map_positions[:-1] + map_positions[1:])
    fine = np.linspace(0, EVALUATION_LENGTH_MB, 8_000)
    truth = _bin_mean(
        fine,
        np.interp(fine, map_centers, archive["mrate"] * RATE_SCALE),
        edges,
    )

    order = np.argsort(archive["fc"])
    positions = archive["fc"][order] / 1e6
    estimate = _bin_mean(positions, archive["fr"][order] * RATE_SCALE, edges)
    lower = _bin_mean(positions, archive["flo"][order] * RATE_SCALE, edges)
    upper = _bin_mean(positions, archive["fhi"][order] * RATE_SCALE, edges)
    return edges, centers, truth, estimate, lower, upper


def load_animation_data() -> tuple[np.ndarray, ...]:
    """Load one contiguous 10 Mb segment of the Drosophila benchmark."""
    archive = np.load(ANIMATION_SOURCE, allow_pickle=True)
    positions = archive["centers_hi_mb"]
    selected = (positions >= 0) & (positions < ANIMATION_LENGTH_MB)
    return (
        positions[selected],
        archive["truth_hi"][selected] * RATE_SCALE,
        archive["fastrho_hi"][selected] * RATE_SCALE,
    )


def evaluation_metrics(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, float, float]:
    """Return shape correlation, median scale ratio, and log10 RMSE."""
    valid = np.isfinite(truth) & np.isfinite(estimate) & (truth > 0) & (estimate > 0)
    pearson = float(np.corrcoef(truth[valid], estimate[valid])[0, 1])
    scale_ratio = float(np.median(estimate[valid] / truth[valid]))
    log10_rmse = float(
        np.sqrt(np.mean((np.log10(estimate[valid]) - np.log10(truth[valid])) ** 2))
    )
    return pearson, scale_ratio, log10_rmse


def make_evaluation_figure(path: Path = EVALUATION_OUTPUT) -> None:
    """Render a static known-map evaluation directly beneath the docs example."""
    _, centers, truth, estimate, lower, upper = load_evaluation_data()
    pearson, scale_ratio, log10_rmse = evaluation_metrics(truth, estimate)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig = plt.figure(figsize=(7.6, 4.2), dpi=120, facecolor=BACKGROUND)
    ax = fig.add_axes([0.095, 0.19, 0.82, 0.57], facecolor=BACKGROUND)
    fig.text(
        0.095,
        0.90,
        "Held-out msprime evaluation",
        color=INK,
        fontsize=14,
        fontweight="normal",
        va="top",
    )
    fig.text(
        0.095,
        0.825,
        (
            f"shape Pearson r = {pearson:.2f}    ·    "
            f"median estimated / true = {scale_ratio:.2f}    ·    "
            f"log10 RMSE = {log10_rmse:.2f}"
        ),
        color=MUTED,
        fontsize=8.5,
        va="top",
    )
    fig.text(
        0.095,
        0.07,
        "Single held-out region  ·  deCODE-derived simulated landscape  ·  25 kb intervals",
        color=MUTED,
        fontsize=8.2,
    )

    transform = lambda values: np.sqrt(np.clip(values, 0, None))  # noqa: E731
    raw_ticks = np.array([0, 1, 4, 9, 16, 25], dtype=float)
    raw_max = max(float(np.nanmax(truth)), float(np.nanmax(upper)))
    y_max = float(transform(raw_max)) * 1.05

    ax.fill_between(
        centers,
        transform(lower),
        transform(upper),
        color=SKY,
        alpha=0.28,
        linewidth=0,
        label="Mean conditional limits",
        zorder=1,
    )
    ax.step(
        centers,
        transform(truth),
        where="mid",
        color=TRUTH,
        linewidth=1.35,
        label="Simulated truth",
        zorder=2,
    )
    ax.plot(
        centers,
        transform(estimate),
        color=BLUE,
        linewidth=2.25,
        solid_capstyle="round",
        label="fastrho",
        zorder=3,
    )
    ax.set_xlim(0, EVALUATION_LENGTH_MB)
    ax.set_ylim(0, y_max)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.72)
    ax.set_axisbelow(True)
    ax.set_xlabel("Position (Mb)", labelpad=8)
    ax.set_ylabel("Rate (cM/Mb; square-root scale)", labelpad=10)
    shown_ticks = raw_ticks[transform(raw_ticks) <= y_max]
    ax.set_yticks(transform(shown_ticks), [f"{x:g}" for x in shown_ticks])
    ax.tick_params(length=0, labelsize=8.2)
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=8.1,
        handlelength=2.1,
        columnspacing=1.25,
        borderaxespad=0.3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, facecolor=BACKGROUND)
    plt.close(fig)
    print(f"wrote {path}")


def make_animation(path: Path = OUTPUT) -> None:
    centers, truth, estimate = load_animation_data()
    agreement = _rank_correlation(truth, estimate)

    # Schematic SNP positions let the animation explain the token/chunk mechanics
    # without implying that 25-kb display bins are the model's native input. Contexts
    # use the same 25% token overlap as ``predict_from_tokens``.
    rng = np.random.default_rng(29)
    gaps = rng.gamma(shape=1.6, scale=1.0, size=130)
    token_positions = 0.06 + (ANIMATION_LENGTH_MB - 0.12) * np.cumsum(gaps) / np.sum(gaps)
    context_tokens = 40
    overlap_tokens = context_tokens // 4
    stride = context_tokens - overlap_tokens
    starts = list(range(0, token_positions.size - context_tokens + 1, stride))
    final_start = token_positions.size - context_tokens
    if starts[-1] != final_start:
        starts.append(final_start)
    contexts = [
        (
            max(0, token_positions[start] - 0.08),
            min(ANIMATION_LENGTH_MB, token_positions[start + context_tokens - 1] + 0.08),
        )
        for start in starts
    ]
    context_masks = [(centers >= left) & (centers <= right) for left, right in contexts]
    overlaps = [
        (max(contexts[i - 1][0], contexts[i][0]), min(contexts[i - 1][1], contexts[i][1]))
        for i in range(1, len(contexts))
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig = plt.figure(figsize=(7.6, 4.2), dpi=100, facecolor=BACKGROUND)
    ax = fig.add_axes([0.095, 0.17, 0.82, 0.42], facecolor=BACKGROUND)
    flow = fig.add_axes([0.095, 0.64, 0.82, 0.16], facecolor=BACKGROUND)

    fig.text(
        0.095,
        0.92,
        "From bidirectional context to one map",
        color=INK,
        fontsize=15,
        fontweight="normal",
        va="top",
    )
    fig.text(
        0.095,
        0.842,
        "Overlapping SNP-token contexts are evaluated in both directions and blended",
        color=MUTED,
        fontsize=8.5,
        va="top",
    )
    fig.text(
        0.095,
        0.025,
        "Schematic token contexts  ·  10 Mb held-out benchmark  ·  Comeron chromosome 2L  ·  25 kb display",
        color=MUTED,
        fontsize=8.2,
    )

    raw_max = max(float(np.nanmax(truth)), float(np.nanmax(estimate)))
    y_max = np.ceil(raw_max / 2) * 2
    raw_ticks = np.arange(0, y_max + 0.1, 4)

    legend_handles = [
        Line2D(
            [],
            [],
            color=TRUTH,
            lw=1.4,
            linestyle=(0, (3, 2)),
            label="Simulated truth",
        ),
        Line2D([], [], color=BLUE, lw=2.2, label="fastrho estimate"),
    ]

    ease = lambda value: value * value * (3 - 2 * value)  # noqa: E731

    def draw(frame: tuple[str, int, float]):
        phase, active, progress = frame
        ax.clear()
        ax.set_facecolor(BACKGROUND)
        ax.set_xlim(0, ANIMATION_LENGTH_MB)
        ax.set_ylim(0, y_max)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.72)
        ax.set_axisbelow(True)
        ax.set_xlabel("Position (Mb)", labelpad=7)
        ax.set_ylabel("Rate (cM/Mb)", labelpad=10)
        ax.set_xticks(np.arange(0, ANIMATION_LENGTH_MB + 0.1, 2))
        ax.set_yticks(raw_ticks)
        ax.tick_params(length=0, labelsize=8.2)

        if phase in {"scan", "project", "blend"}:
            for index in range(active):
                selected = context_masks[index]
                ax.plot(
                    centers[selected],
                    estimate[selected],
                    color=SKY,
                    linewidth=1.7,
                    solid_capstyle="round",
                    zorder=3,
                )

            if phase in {"project", "blend"}:
                selected = context_masks[active]
                vertical_scale = ease(progress) if phase == "project" else 1.0
                ax.plot(
                    centers[selected],
                    estimate[selected] * vertical_scale,
                    color=SKY,
                    linewidth=1.7,
                    solid_capstyle="round",
                    zorder=3,
                )

            if phase == "scan":
                left, right = contexts[active]
                ax.axvspan(left, right, color=SKY, alpha=0.065, linewidth=0, zorder=1)
                status = f"context {active + 1}/{len(contexts)} · read both directions"
            elif phase == "project":
                status = f"context {active + 1}/{len(contexts)} · project all intervals"
            else:
                overlap_left, overlap_right = overlaps[active - 1]
                ax.axvspan(
                    overlap_left,
                    overlap_right,
                    facecolor=SKY,
                    edgecolor=BLUE,
                    hatch="////",
                    alpha=0.18,
                    linewidth=0,
                    zorder=1,
                )
                status = "Hann-weighted overlap"
        elif phase in {"stitch", "evaluate", "complete"}:
            ax.plot(
                centers,
                estimate,
                color=BLUE,
                linewidth=1.2 + 0.6 * ease(progress) if phase == "stitch" else 1.8,
                solid_capstyle="round",
                label="fastrho estimate",
                zorder=3,
            )
            if phase in {"evaluate", "complete"}:
                ax.step(
                    centers,
                    truth,
                    where="mid",
                    color=TRUTH,
                    linewidth=1.1,
                    linestyle=(0, (3, 2)),
                    label="Simulated truth",
                    zorder=2,
                )
            if phase == "stitch":
                status = "lock context outputs into one map"
            elif phase == "evaluate":
                status = "evaluation overlay · simulated truth"
            else:
                status = f"held-out rank correlation {agreement:.2f}"
        else:
            status = "one feature token per SNP"
        ax.text(
            0.985,
            0.92,
            status,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=BLUE if phase not in {"input", "evaluate"} else MUTED,
            fontsize=8.1,
            fontweight="normal",
            zorder=8,
        )
        if phase in {"evaluate", "complete"}:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                ncol=2,
                frameon=False,
                fontsize=8.1,
                handlelength=2.1,
                columnspacing=1.25,
                borderaxespad=0.3,
            )

        flow.clear()
        flow.set_xlim(0, ANIMATION_LENGTH_MB)
        flow.set_ylim(0, 1)
        flow.axis("off")
        token_height = 0.14 * ease(progress) if phase == "input" else 0.14
        flow.plot(
            [0, ANIMATION_LENGTH_MB],
            [0.71, 0.71],
            color=GRID,
            linewidth=1,
            zorder=1,
        )
        flow.vlines(
            token_positions,
            0.71 - token_height / 2,
            0.71 + token_height / 2,
            color=INK,
            linewidth=0.7,
            alpha=0.85,
            zorder=2,
        )
        flow.text(0, 0.93, "SNP tokens", color=MUTED, fontsize=7.8, va="top")

        if phase in {"scan", "project"}:
            left, right = contexts[active]
            window = FancyBboxPatch(
                (left, 0.08),
                right - left,
                0.47,
                boxstyle="round,pad=0.02,rounding_size=0.07",
                facecolor=SKY,
                edgecolor=BLUE,
                linewidth=0.8,
                alpha=0.16,
                zorder=1,
            )
            flow.add_patch(window)
            span = right - left
            arrow_progress = max(0.04, progress) if phase == "scan" else 1.0
            flow.annotate(
                "",
                xy=(left + arrow_progress * span, 0.42),
                xytext=(left + 0.04 * span, 0.42),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.25},
                zorder=4,
            )
            flow.annotate(
                "",
                xy=(right - arrow_progress * span, 0.22),
                xytext=(right - 0.04 * span, 0.22),
                arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.25},
                zorder=4,
            )
            flow.text(
                0,
                0.01,
                (
                    f"bidirectional Mamba context {active + 1}/{len(contexts)}"
                    if phase == "scan"
                    else "project every interval in this context at once"
                ),
                color=MUTED,
                fontsize=7.8,
                va="bottom",
            )
            flow.text(
                ANIMATION_LENGTH_MB,
                0.01,
                "context output" if phase == "project" else "forward + reverse sequence mixing",
                color=MUTED,
                fontsize=7.8,
                ha="right",
                va="bottom",
            )
        elif phase == "blend":
            previous_left, previous_right = contexts[active - 1]
            current_left, current_right = contexts[active]
            for left, right, edge in [
                (previous_left, previous_right, MUTED),
                (current_left, current_right, BLUE),
            ]:
                flow.add_patch(
                    FancyBboxPatch(
                        (left, 0.13),
                        right - left,
                        0.40,
                        boxstyle="round,pad=0.02,rounding_size=0.07",
                        facecolor=BACKGROUND,
                        edgecolor=edge,
                        linewidth=0.8,
                        zorder=1,
                    )
                )
            overlap_left, overlap_right = overlaps[active - 1]
            flow.axvspan(
                overlap_left,
                overlap_right,
                ymin=0.13,
                ymax=0.53,
                facecolor=SKY,
                edgecolor=BLUE,
                hatch="////",
                alpha=0.28,
                linewidth=0,
                zorder=2,
            )
            flow.text(0, 0.01, "Hann-weighted overlap", color=MUTED, fontsize=7.8)
            flow.text(
                ANIMATION_LENGTH_MB,
                0.01,
                "combine predictive moments",
                color=MUTED,
                fontsize=7.8,
                ha="right",
            )
        elif phase == "stitch":
            stitch_progress = ease(progress)
            for index, (left, right) in enumerate(contexts):
                initial_y = 0.16 + 0.10 * (index % 3)
                y = initial_y * (1 - stitch_progress) + 0.30 * stitch_progress
                flow.plot(
                    [left, right],
                    [y, y],
                    color=BLUE,
                    linewidth=4.5,
                    solid_capstyle="round",
                )
            flow.text(0, 0.01, "lock context edges", color=MUTED, fontsize=7.8)
            flow.text(
                ANIMATION_LENGTH_MB,
                0.01,
                "one continuous interval map",
                color=MUTED,
                fontsize=7.8,
                ha="right",
            )
        elif phase in {"evaluate", "complete"}:
            flow.plot(
                [0, ANIMATION_LENGTH_MB],
                [0.30, 0.30],
                color=BLUE,
                linewidth=5,
                solid_capstyle="round",
                alpha=0.75,
            )
            flow.text(0, 0.08, "stitched interval predictions", color=MUTED, fontsize=7.8)
            if phase == "evaluate":
                flow.text(
                    ANIMATION_LENGTH_MB,
                    0.08,
                    "truth added for evaluation only",
                    color=TRUTH,
                    fontsize=7.8,
                    ha="right",
                )
        return []

    frames: list[tuple[str, int, float]] = []
    frames.extend(("input", -1, float(p)) for p in np.linspace(0.08, 1, 5))
    for index in range(len(contexts)):
        frames.extend(("scan", index, float(p)) for p in np.linspace(0.08, 1, 6))
        frames.extend(("project", index, float(p)) for p in np.linspace(0.08, 1, 5))
        if index > 0:
            frames.extend([("blend", index, 1.0)] * 3)
    frames.extend(("stitch", len(contexts), float(p)) for p in np.linspace(0.05, 1, 5))
    frames.extend([("evaluate", len(contexts), 1.0)] * 7)
    frames.extend([("complete", len(contexts), 1.0)] * 14)
    animation = FuncAnimation(fig, draw, frames=frames, interval=95, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PlayOncePillowWriter(fps=11))
    plt.close(fig)
    print(f"wrote {path}")


def make_architecture_animation(path: Path = OUTPUT) -> None:
    """Animate the native BiMamba computation at SNP-token resolution.

    The original hero emphasized chunk stitching and a 25-kb evaluation grid.
    This version instead expands the paper architecture: the multi-scale stem,
    two Mamba-2 scans, per-token merge, residual/MLP path, encoder-decoder stack,
    and one output distribution per adjacent-SNP interval.
    """
    _, _, benchmark_estimate = load_animation_data()

    rng = np.random.default_rng(29)
    n_tokens = 88
    gaps = rng.gamma(shape=2.2, scale=1.0, size=n_tokens)
    token_x = 0.105 + 0.79 * np.cumsum(gaps) / np.sum(gaps)
    interval_x = 0.5 * (token_x[:-1] + token_x[1:])

    # A committed benchmark supplies a plausible fine-scale silhouette.  The
    # schematic remains unitless because its purpose is model topology and
    # native resolution, not quantitative evaluation.
    source_x = np.linspace(0, 1, benchmark_estimate.size)
    sampled = np.interp(np.linspace(0, 0.24, interval_x.size), source_x, benchmark_estimate)
    sampled = np.nan_to_num(sampled, nan=float(np.nanmedian(benchmark_estimate)))
    low, high = np.percentile(sampled, [5, 95])
    scaled = np.clip((sampled - low) / max(high - low, 1e-8), 0, 1)
    output_y = 0.145 + 0.055 * scaled
    output_band = 0.009 + 0.006 * (1 - scaled)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig = plt.figure(figsize=(7.6, 4.7), dpi=110, facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1], facecolor=BACKGROUND)

    pipeline = [
        (0.035, 0.765, 0.145, 0.092, "SNP features", "17–18 channels"),
        (0.225, 0.765, 0.145, 0.092, "multi-scale stem", "conv k = 3, 7, 15"),
        (0.415, 0.765, 0.145, 0.092, "encoder", "6 × BiMamba"),
        (0.605, 0.765, 0.145, 0.092, "decoder", "4 × BiMamba + skips"),
        (0.795, 0.765, 0.170, 0.092, "interval head", "mean + log variance"),
    ]
    stage_order = {
        "input": 0,
        "stem": 1,
        "scan": 2,
        "merge": 2,
        "block": 2,
        "decode": 3,
        "output": 4,
        "complete": 4,
    }
    ease = lambda value: value * value * (3 - 2 * value)  # noqa: E731

    def draw(frame: tuple[str, int, float]):
        phase, _, progress = frame
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(BACKGROUND)

        ax.text(
            0.035,
            0.945,
            "How fastrho reads both sides of every SNP",
            color=INK,
            fontsize=15,
            va="top",
        )
        ax.text(
            0.035,
            0.895,
            "The paper architecture, animated at the model's native token-to-interval resolution",
            color=MUTED,
            fontsize=8.5,
            va="top",
        )

        current_stage = stage_order[phase]
        for index, (x, y, width, height, title, detail) in enumerate(pipeline):
            is_active = index == current_stage
            is_complete = index < current_stage
            face = "#EAF6FB" if is_active else ("#F1F8F5" if is_complete else "#FFFFFF")
            edge = BLUE if is_active else (GREEN if is_complete else "#BFCBD3")
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    width,
                    height,
                    boxstyle="round,pad=0.008,rounding_size=0.012",
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.25 if is_active else 0.75,
                    zorder=2,
                )
            )
            ax.text(
                x + width / 2,
                y + 0.059,
                title,
                ha="center",
                va="center",
                color=INK,
                fontsize=8.0,
            )
            ax.text(
                x + width / 2,
                y + 0.027,
                detail,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=6.5,
            )
            if index < len(pipeline) - 1:
                next_x = pipeline[index + 1][0]
                ax.annotate(
                    "",
                    xy=(next_x - 0.008, y + height / 2),
                    xytext=(x + width + 0.008, y + height / 2),
                    arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.75},
                )

        ax.add_patch(
            FancyBboxPatch(
                (0.245, 0.693),
                0.285,
                0.031,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor="#FFF8E8",
                edgecolor=ORANGE,
                linewidth=0.55,
            )
        )
        ax.text(
            0.387,
            0.708,
            "FiLM · mutation rate · sample size · input view",
            color=ORANGE,
            fontsize=5.9,
            ha="center",
            va="center",
        )
        ax.annotate(
            "",
            xy=(0.447, 0.765),
            xytext=(0.447, 0.725),
            arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 0.7},
        )
        ax.annotate(
            "encoder skip states",
            xy=(0.655, 0.765),
            xytext=(0.615, 0.707),
            color=MUTED,
            fontsize=5.9,
            ha="center",
            arrowprops={
                "arrowstyle": "-|>",
                "color": MUTED,
                "lw": 0.65,
                "connectionstyle": "arc3,rad=-0.18",
            },
        )

        ax.add_patch(
            FancyBboxPatch(
                (0.035, 0.105),
                0.93,
                0.57,
                boxstyle="round,pad=0.012,rounding_size=0.016",
                facecolor="#FFFFFF",
                edgecolor="#B8D8E9",
                linewidth=0.9,
                linestyle=(0, (3, 2)),
                zorder=0,
            )
        )
        ax.text(
            0.055,
            0.647,
            "one pre-normalized BiMamba block — expanded",
            color=BLUE,
            fontsize=8.0,
            va="top",
        )
        ax.text(
            0.945,
            0.647,
            "LN → two Mamba-2 scans → concat + project → residual → LN → MLP → residual",
            color=MUTED,
            fontsize=6.5,
            ha="right",
            va="top",
        )

        reveal = n_tokens if phase != "input" else max(1, int(ease(progress) * n_tokens))
        ax.plot([token_x[0], token_x[-1]], [0.555, 0.555], color=GRID, lw=0.9, zorder=1)
        ax.vlines(
            token_x[:reveal],
            0.544,
            0.566,
            color=INK,
            linewidth=0.65,
            alpha=0.88,
            zorder=3,
        )
        ax.text(0.055, 0.555, "SNP tokens", color=MUTED, fontsize=6.7, va="center")
        ax.text(
            0.945,
            0.585,
            "one mark = one retained SNP",
            color=MUTED,
            fontsize=6.5,
            va="center",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
        )

        if phase == "stem":
            focal = n_tokens // 2
            kernel_progress = ease(progress)
            for rank, (kernel, color) in enumerate([(15, SKY), (7, BLUE), (3, PURPLE)]):
                radius = (kernel - 1) // 2
                left = token_x[max(0, focal - radius)]
                right = token_x[min(n_tokens - 1, focal + radius)]
                half = 0.5 * (right - left) * kernel_progress
                center = token_x[focal]
                y = 0.515 - rank * 0.021
                ax.plot(
                    [center - half, center + half],
                    [y, y],
                    color=color,
                    lw=3.0 - 0.45 * rank,
                    solid_capstyle="round",
                    alpha=0.78,
                )
                ax.text(
                    center + half + 0.008,
                    y,
                    f"k={kernel}",
                    color=MUTED,
                    fontsize=6.0,
                    va="center",
                )
            ax.text(0.055, 0.493, "local stem", color=MUTED, fontsize=6.7, va="center")

        show_scans = phase in {"scan", "merge", "block", "decode", "output", "complete"}
        scan_progress = ease(progress) if phase == "scan" else 1.0
        if show_scans:
            scanned = max(2, int(scan_progress * n_tokens))
            forward_x = token_x[:scanned]
            reverse_x = token_x[-scanned:]
            ax.plot([token_x[0], token_x[-1]], [0.430, 0.430], color=GRID, lw=0.75)
            ax.plot([token_x[0], token_x[-1]], [0.350, 0.350], color=GRID, lw=0.75)
            ax.plot(
                forward_x,
                np.full_like(forward_x, 0.430),
                color=BLUE,
                lw=2.1,
                solid_capstyle="round",
                zorder=3,
            )
            ax.plot(
                reverse_x,
                np.full_like(reverse_x, 0.350),
                color=ORANGE,
                lw=2.1,
                solid_capstyle="round",
                zorder=3,
            )
            ax.scatter(forward_x, np.full_like(forward_x, 0.430), s=5.5, color=BLUE, zorder=4)
            ax.scatter(reverse_x, np.full_like(reverse_x, 0.350), s=5.5, color=ORANGE, zorder=4)
            ax.annotate(
                "",
                xy=(forward_x[-1], 0.430),
                xytext=(forward_x[max(0, len(forward_x) - 5)], 0.430),
                arrowprops={"arrowstyle": "-|>", "color": BLUE, "lw": 1.05},
            )
            ax.annotate(
                "",
                xy=(reverse_x[0], 0.350),
                xytext=(reverse_x[min(4, len(reverse_x) - 1)], 0.350),
                arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.05},
            )
            ax.text(0.055, 0.430, "forward", color=BLUE, fontsize=6.7, va="center")
            ax.text(0.055, 0.350, "reverse", color=ORANGE, fontsize=6.7, va="center")
            ax.text(
                0.945,
                0.390,
                "selective state carries context through every SNP",
                color=MUTED,
                fontsize=6.5,
                ha="right",
                va="center",
            )

        if phase in {"merge", "block", "decode", "output", "complete"}:
            merge_progress = ease(progress) if phase == "merge" else 1.0
            merged = max(1, int(merge_progress * n_tokens))
            merged_x = token_x[:merged]
            ax.vlines(
                merged_x,
                0.275,
                0.420,
                color=PURPLE,
                linewidth=0.35,
                alpha=0.22,
                zorder=1,
            )
            ax.scatter(merged_x, np.full_like(merged_x, 0.270), s=7.0, color=PURPLE, zorder=4)
            ax.text(0.055, 0.270, "merge", color=PURPLE, fontsize=6.7, va="center")
            ax.text(
                0.945,
                0.296,
                "concat forward + reverse states, then project per token",
                color=MUTED,
                fontsize=6.5,
                ha="right",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
            )

        if phase in {"block", "decode", "output", "complete"}:
            block_progress = ease(progress) if phase == "block" else 1.0
            labels = ["+ scan residual", "LN → 4× channel MLP", "+ MLP residual"]
            box_width = 0.162
            start = 0.235
            visible = max(1, int(np.ceil(block_progress * len(labels))))
            for index, label in enumerate(labels[:visible]):
                x = start + index * 0.185
                ax.add_patch(
                    FancyBboxPatch(
                        (x, 0.217),
                        box_width,
                        0.042,
                        boxstyle="round,pad=0.005,rounding_size=0.009",
                        facecolor="#F5F0FA" if index == 1 else "#F7FAFC",
                        edgecolor=PURPLE if index == 1 else "#BFCBD3",
                        linewidth=0.7,
                    )
                )
                ax.text(
                    x + box_width / 2,
                    0.238,
                    label,
                    ha="center",
                    va="center",
                    color=INK,
                    fontsize=6.1,
                )
                if index < visible - 1:
                    ax.annotate(
                        "",
                        xy=(x + 0.181, 0.238),
                        xytext=(x + box_width + 0.004, 0.238),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 0.6},
                    )

        if phase == "decode":
            copies = max(1, int(np.ceil(ease(progress) * 4)))
            for index in range(copies):
                ax.add_patch(
                    FancyBboxPatch(
                        (0.745 + 0.009 * index, 0.205 + 0.009 * index),
                        0.115,
                        0.055,
                        boxstyle="round,pad=0.004,rounding_size=0.008",
                        facecolor="#EAF6FB",
                        edgecolor=BLUE,
                        linewidth=0.65,
                        alpha=0.9,
                    )
                )
            ax.text(
                0.815,
                0.183,
                "decoder repeats the same bidirectional block",
                color=MUTED,
                fontsize=6.2,
                ha="center",
            )

        if phase in {"output", "complete"}:
            output_progress = ease(progress) if phase == "output" else 1.0
            n_intervals = max(2, int(output_progress * interval_x.size))
            shown_x = interval_x[:n_intervals]
            shown_y = output_y[:n_intervals]
            shown_band = output_band[:n_intervals]
            ax.fill_between(
                shown_x,
                shown_y - shown_band,
                shown_y + shown_band,
                color=SKY,
                alpha=0.32,
                linewidth=0,
                zorder=1,
            )
            ax.plot(shown_x, shown_y, color=BLUE, linewidth=1.55, zorder=3)
            ax.vlines(
                token_x[: n_intervals + 1],
                0.122,
                0.132,
                color=INK,
                linewidth=0.42,
                alpha=0.48,
            )
            ax.text(0.055, 0.155, "interval output", color=BLUE, fontsize=6.7, va="center")
            ax.text(
                0.945,
                0.155,
                "mean and variance for every adjacent-SNP interval",
                color=MUTED,
                fontsize=6.5,
                ha="right",
                va="center",
            )

        status = {
            "input": "embed one feature vector per retained SNP",
            "stem": "mix local patterns at three kernel widths",
            "scan": "run independent Mamba-2 state scans in both directions",
            "merge": "join left and right context at every token",
            "block": "complete the two residual paths",
            "decode": "repeat through six encoder and four decoder blocks",
            "output": "predict every interval—not a coarse genomic bin",
            "complete": "fine-scale bidirectional state-space inference",
        }[phase]
        ax.text(0.50, 0.078, status, color=BLUE, fontsize=7.1, ha="center")
        ax.text(
            0.035,
            0.035,
            "Schematic 1,024-token context  ·  horizontal spacing is token order  ·  native output is one adjacent-SNP interval",
            color=MUTED,
            fontsize=6.7,
            va="bottom",
        )
        return []

    frames: list[tuple[str, int, float]] = []
    frames.extend(("input", 0, float(p)) for p in np.linspace(0.04, 1, 8))
    frames.extend(("stem", 0, float(p)) for p in np.linspace(0.05, 1, 10))
    frames.extend(("scan", 0, float(p)) for p in np.linspace(0.03, 1, 22))
    frames.extend(("merge", 0, float(p)) for p in np.linspace(0.03, 1, 12))
    frames.extend(("block", 0, float(p)) for p in np.linspace(0.05, 1, 9))
    frames.extend(("decode", 0, float(p)) for p in np.linspace(0.05, 1, 8))
    frames.extend(("output", 0, float(p)) for p in np.linspace(0.03, 1, 16))
    frames.extend([("complete", 0, 1.0)] * 18)

    animation = FuncAnimation(fig, draw, frames=frames, interval=84, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PlayOncePillowWriter(fps=8))
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    make_evaluation_figure()
    make_architecture_animation()
