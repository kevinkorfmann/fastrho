"""fig_treeoflife_panel.pdf — cross-species population recombination landscapes.
Per species: a black organism silhouette (PhyloPic, stored in paper/figdata/silhouettes/
<key>.png), the common name, and the fastrho-recovered recombination landscape
(cobalt line; dashed ink = published map where one exists). No phylogeny, no big subheader.
Columns grow with species count. One frozen model, unphased genotypes, one forward pass each.

Run: PYTHONNOUSERSITE=1 python scripts/fig_treeoflife_panel.py
"""
import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001 -- backend must be selected before pyplot import
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

import paper_style as ps

ps.style()
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDATA = os.path.join(_HERE, "paper", "figdata")
SILH = os.path.join(FIGDATA, "silhouettes")
OUT = os.path.join(_HERE, "paper", "figures")

CLADE_ORDER = ["Mammals", "Primates", "Birds", "Reptiles", "Amphibians", "Fish", "Molluscs",
               "Arthropods", "Insects", "Nematodes", "Cnidaria", "Other", "Plants", "Fungi"]
INK = "#151515"
FAST_COLOR = ps.C["fastrho"]
PYRHO_COLOR = ps.C["pyrho"]


def norm(a):
    a = np.asarray(a, float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo + 1e-12)


def main():
    data = json.load(open(os.path.join(FIGDATA, "transect.json")))
    pyr_path = os.path.join(FIGDATA, "transect_pyrho.json")
    pyr = json.load(open(pyr_path)) if os.path.exists(pyr_path) else {}
    sp = [e for e in data["species"] if e.get("track") and e["track"].get("pred")]
    sp.sort(key=lambda e: (CLADE_ORDER.index(e["clade"]) if e["clade"] in CLADE_ORDER else 98,
                           e["common"]))
    missing_silhouettes = [
        e["key"] for e in sp if not os.path.isfile(os.path.join(SILH, f"{e['key']}.png"))
    ]
    if missing_silhouettes:
        raise FileNotFoundError(
            "Missing required silhouettes for retained species: " + ", ".join(missing_silhouettes)
        )
    n = len(sp)
    ncol = 1 if n <= 8 else 2
    nrow = math.ceil(n / ncol)

    cellw, cellh = 5.0, 0.62
    fig = plt.figure(figsize=(cellw * ncol, cellh * nrow + 0.9), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ncol)
    ax.set_ylim(nrow + 0.7, -0.6)
    ax.axis("off")

    # column-major fill, clade-contiguous
    for i, e in enumerate(sp):
        c = i // nrow
        r = i % nrow
        x0 = c + 0.03
        y = r + 0.5
        cc = FAST_COLOR
        # --- name (far left); the silhouette now sits in the map's top-right, freeing this column ---
        display_name = e["common"] + ("†" if e.get("qualification_tier") == "context-limited" else "")
        ax.text(x0 + 0.01, y - 0.10, display_name, fontsize=8.6, va="center", ha="left",
                color=INK, fontweight="medium")
        if e.get("latin"):
            ax.text(x0 + 0.01, y + 0.17, e["latin"], fontsize=5.8, va="center", ha="left",
                    style="italic", color="#8a949c")
        # --- landscape map (dominant, WIDER: starts right after the name) ---
        mx0, mx1 = x0 + (0.28 if ncol > 1 else 0.17), c + 0.885
        tr = e["track"]
        xx = np.linspace(mx0, mx1, len(tr["pred"]))
        base = y + 0.30
        if tr.get("truth") and e.get("validated"):
            ax.plot(
                xx,
                base - norm(tr["truth"]) * 0.58,
                color=ps.C["truth"],
                lw=0.75,
                ls=(0, (4, 2.5)),
                zorder=2,
            )
        pv = base - norm(tr["pred"]) * 0.58
        ax.plot(xx, pv, color=cc, lw=1.2, zorder=4)
        # --- pyrho (composite-likelihood gold standard, genotype mode) as a thin overlay ---
        pe = pyr.get(e["key"], {})
        pyv = pe.get("pyrho")
        if pyv and any(v is not None for v in pyv):
            pa = np.array([np.nan if v is None else v for v in pyv], float)
            pn = norm(pa)
            # light 3-point smoothing (display only) so the denser panels read as a shape, not noise
            if len(pn) >= 5:
                kern = np.array([0.25, 0.5, 0.25])
                pn = np.convolve(np.nan_to_num(pn, nan=np.nanmean(pn)), kern, mode="same")
            k2 = min(len(pn), len(xx))
            ax.plot(
                xx[:k2],
                base - pn[:k2] * 0.58,
                color=PYRHO_COLOR,
                lw=0.65,
                ls=(0, (2, 2)),
                alpha=0.85,
                zorder=5,
            )
        # --- silhouette overlaid at the map's TOP-RIGHT corner (a subtle label, not a column) ---
        sp_png = os.path.join(SILH, f"{e['key']}.png")
        try:
            img = plt.imread(sp_png)
        except Exception as exc:
            raise RuntimeError(f"Could not read required silhouette: {sp_png}") from exc
        if img.ndim == 3 and img.shape[2] == 4:
            rgba = np.zeros_like(img)
            rgba[..., 0] = 0.15
            rgba[..., 1] = 0.19
            rgba[..., 2] = 0.23
            rgba[..., 3] = img[..., 3] * 0.88
            img = rgba
        zoom = 27.0 / max(img.shape[0], img.shape[1], 1)
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (mx1 - 0.04, base - 0.50),
                            frameon=False, box_alignment=(0.5, 0.5), pad=0, zorder=6)
        ax.add_artist(ab)
        # accuracy metric: r vs published map (validated) else faint reproducibility rho.
        # extreme regimes (bottleneck / selfing) carry a specialist tag; a selfer whose population
        # LD inverts the meiotic map is the SWEEP signature, not a failure -> slate, labelled so.
        r_ = e.get("pearson")
        lg = e.get("log_reproducibility")
        regime = e.get("regime", "outbred")
        snote = e.get("specialist_note") or ""
        bx = c + 0.905
        SPEC = {"canid specialist (wide LD radii)": "canid model",
                "selfing-scaled specialist": "selfing model"}
        if r_ is not None:
            mc = FAST_COLOR if r_ >= 0 else "#666666"
            ax.text(bx, y - 0.10, f"r={r_:+.2f}", fontsize=7.6, va="center", ha="left",
                    color=mc, fontweight="bold")
            if regime == "selfing" and r_ < 0:
                sub = "selfing sweep"
            elif regime != "outbred" and snote:
                sub = SPEC.get(snote, snote.split(" (")[0])
            else:
                sub = "vs map"
            ax.text(bx, y + 0.17, sub, fontsize=5.0, va="center", ha="left", color="#9aa4ac")
        elif lg is not None:
            ax.text(bx, y - 0.10, f"ρ={lg:.2f}", fontsize=7.0, va="center", ha="left",
                    color="#8a949c")
            ax.text(bx, y + 0.17, "split repeat.", fontsize=5.0, va="center", ha="left",
                    color="#b0b8be")
        # fastrho-vs-pyrho concordance (small): how well the independent estimator agrees
        cr = pyr.get(e["key"], {}).get("concordance_r")
        if cr is not None:
            ax.text(bx, y + 0.30, f"vs pyrho {cr:+.2f}", fontsize=5.0, va="center", ha="left",
                    color=PYRHO_COLOR)

    # title only (non-bold); the explanatory text lives in the paper figure legend, not on the plate
    ax.text(0.03, -0.28, "Cross-species recombination landscapes", fontsize=13.5,
            fontweight="normal", color=INK, ha="left", va="center")
    method_handles = [
        Line2D([0], [0], color=FAST_COLOR, lw=1.6, label="fastrho"),
        Line2D([0], [0], color=ps.C["truth"], lw=1.0, ls=(0, (4, 2.5)),
               label="published map"),
        Line2D([0], [0], color=PYRHO_COLOR, lw=0.9, ls=(0, (2, 2)), label="pyrho"),
    ]
    fig.legend(handles=method_handles, loc="upper right", bbox_to_anchor=(0.985, 0.985),
               ncol=3, frameon=False, fontsize=7.0, handlelength=1.5,
               columnspacing=1.1, handletextpad=0.45)
    ps.save(fig, "fig_treeoflife_panel", outdir=OUT, formats=("pdf",))
    fig.savefig(os.path.join(OUT, "fig_treeoflife_panel.png"), dpi=185, bbox_inches="tight",
                facecolor="white")
    print(f"panel: {n} species, {ncol}x{nrow}")


if __name__ == "__main__":
    main()
