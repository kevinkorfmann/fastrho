"""Shared figure style for the fastrho paper -- single source of truth.

Encodes the locked, colorblind-safe palette and the publication rcParams used by
every figure (paperfig.py + the standalone fig_*.py scripts) so that a colour means
the same method/category in every panel of every figure. Built to the
scientific-visualization skill's publication standards: sans-serif, despined,
colorblind-safe (NO red-green), and editable embedded vector text (Type-42 fonts)
so reviewers/typesetters can edit the PDF.

Usage
-----
    import paper_style as ps
    ps.style()                          # apply rcParams (call once, near the top)
    ...build the figure...
    ps.panel(ax, "a")                   # bold panel letter
    ps.save(fig, "fig_selection")       # -> paper/figures/fig_selection.pdf

The module lives in scripts/, so any script run as ``python scripts/fig_X.py`` can
``import paper_style`` directly (its directory is on sys.path[0]).
"""
from __future__ import annotations

import os

import matplotlib
from cycler import cycler

# --- restrained editorial palette --------------------------------------------------
# One cobalt accent carries the focal method; comparison methods use ink and gray.
# Shape and line style provide the redundant encoding wherever several methods meet.
C = {
    "fastrho":   "#2737E7",
    "fastrho_l": "#6F7CF0",
    "pyrho":     "#4D4D4D",
    "relernn":   "#8A8A8A",
    "truth":     "#151515",
    "gru":       "#B0B0B0",
}
LAB = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN",
       "truth": "true map", "gru": "GRU"}

# Extended categories stay within the same cobalt/ink/gray family.
CB = ["#2737E7", "#151515", "#777777", "#A6A6A6", "#4F5FEB", "#333333",
      "#8A8A8A", "#6F7CF0", "#B8B8B8", "#555555", "#C8C8C8", "#999999"]

# named semantic groups, kept consistent across figures (drawn from Paired)
REGIME = {"neutral": "#8A8A8A", "bgs": "#151515", "sweep": "#2737E7"}
SPECIES = {"human": "#2737E7", "dmel": "#151515", "athal": "#777777", "dog": "#A6A6A6"}
# village->breed transfer (dog figure): village = fastrho-blue (the transferred estimate),
# breed = orange. Deliberately NOT green (#33a02c is the pyrho method colour) so a breed's
# own-data curve is never confused with pyrho. Blue/orange/black is colorblind-safe.
TRANSFER = {"village": C["fastrho"], "breed": C["relernn"]}

# light highlight wash (e.g. inversion / sweep span) -- neutral, low-chroma
HIGHLIGHT = "#F4F4F1"

# perceptually uniform colormaps (per the skill: never jet/rainbow)
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"


def style():
    """Apply the paper's publication rcParams. Idempotent; call once near the top.

    Deliberately does NOT set ``figure.figsize`` -- the paper's figures are authored
    oversized and scaled in LaTeX, so each script keeps its own size.
    """
    matplotlib.rcParams.update({
        # fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "normal",
        "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        # spines / grid
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.85, "axes.axisbelow": True,
        "axes.grid": False, "grid.color": "#D4D4D0", "grid.alpha": 0.35,
        "grid.linewidth": 0.45,
        "xtick.color": "#333333", "ytick.color": "#333333",
        "legend.frameon": False,
        "lines.linewidth": 1.5, "lines.markersize": 5.0,
        # colorblind-safe default cycle
        "axes.prop_cycle": cycler(color=CB),
        # editable, embedded vector text (journal requirement)
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        # save
        "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def panel(ax, letter, x=-0.13, y=1.06, fontsize=14):
    """Bold lower-case panel label at the top-left of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=fontsize,
            fontweight="bold", va="top", ha="left")


def barlabels(ax, fmt="%.2f", fs=7.6):
    """Annotate every positive bar in ``ax`` with its height."""
    for p in ax.patches:
        h = p.get_height()
        if h and h == h and h > 0.001:
            ax.annotate(fmt % h, (p.get_x() + p.get_width() / 2, h), ha="center",
                        va="bottom", fontsize=fs, color="#444", xytext=(0, 1.4),
                        textcoords="offset points")


def _figdir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "paper", "figures")


def save(fig, name, outdir=None, formats=("pdf",), dpi=600):
    """Save ``fig`` to <paper/figures>/<name>.<fmt> at publication DPI."""
    outdir = outdir or _figdir()
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for fmt in formats:
        p = os.path.join(outdir, f"{name}.{fmt}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
        print("wrote", p)
    return paths
