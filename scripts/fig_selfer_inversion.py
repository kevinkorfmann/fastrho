"""Extended Data figure: the pyrho map inversion on a selfer is systematic across the genome.

The main text (Fig. 5c) shows, on real A. thaliana chr1, that a naive LD method (pyrho) recovers a
map that is *anti-correlated* with the truth -- it invents a recombination hotspot in the cold
pericentromere -- while the selfing-aware fastrho model recovers the real landscape. This figure
drives that home: the identical pipeline (extract -> fastrho self2 -> pyrho, same 156 Swedish
accessions, same Salome/TAIR10 map, one shared pyrho lookup table) run on ALL FIVE chromosomes.

(a) Per-chromosome standardized-rate trajectories (z; linear so Pearson r is preserved): true map
    (black), selfing-aware fastrho (blue), pyrho (green). The pericentromeric low-recombination
    windows (truth below its 25th percentile) are shaded -- pyrho systematically peaks there.
(b) The sign flip made explicit: every chromosome's fastrho r is positive and its pyrho r is
    negative. One naive method, inverted on every chromosome of a real selfer genome.

Data: paper/figdata/selfer_chroms.npz (bundle_selfer_chroms.py, on sesame).
Run: PYTHONNOUSERSITE=1 python scripts/fig_selfer_inversion.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(_HERE, "paper", "figdata", "selfer_chroms.npz")
OUT = os.path.join(_HERE, "paper", "figures")
CHROMS = ["1", "2", "3", "4", "5"]

ps.style()


def _zmask(x, m):
    """Standardize over valid windows only (linear -> preserves Pearson r)."""
    x = np.asarray(x, float)
    z = np.full_like(x, np.nan)
    z[m] = (x[m] - x[m].mean()) / x[m].std()
    return z


def _smooth(x, w=5):
    """Moving average that ignores NaNs (keeps gaps from bleeding across invalid windows)."""
    x = np.asarray(x, float)
    v = np.isfinite(x).astype(float)
    xf = np.where(np.isfinite(x), x, 0.0)
    k = np.ones(w)
    num = np.convolve(xf, k, mode="same")
    den = np.convolve(v, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    out[~np.isfinite(x)] = np.nan
    return out


def _pericentromere(cen, ztruth, m, w=11, central=(0.15, 0.85), pct=30):
    """Single contiguous pericentromeric low-recombination block: the deepest central sink of
    the heavily smoothed true map, extended left/right while it stays below the ``pct`` percentile.
    Returns (lo, hi) window indices, or (None, None)."""
    n = len(cen)
    sm = _smooth(ztruth, w)
    good = np.isfinite(sm) & m
    idx = np.arange(n)
    inner = good & (idx >= int(central[0] * n)) & (idx <= int(central[1] * n))
    if not inner.any():
        return None, None
    i0 = idx[inner][np.argmin(sm[inner])]
    thr = np.nanpercentile(sm[good], pct)
    lo = i0
    while lo - 1 >= 0 and np.isfinite(sm[lo - 1]) and sm[lo - 1] < thr:
        lo -= 1
    hi = i0
    while hi + 1 < n and np.isfinite(sm[hi + 1]) and sm[hi + 1] < thr:
        hi += 1
    return lo, hi


def panel_trajectories(fig, cell, d):
    sub = cell.subgridspec(1, len(CHROMS), wspace=0.12)
    axes = []
    for j, c in enumerate(CHROMS):
        ax = fig.add_subplot(sub[j])
        axes.append(ax)
        cen = d[f"c{c}_centers"]
        truth = d[f"c{c}_truth"]; pred = d[f"c{c}_pred"]; pyr = d[f"c{c}_pyrho"]
        m = np.isfinite(truth) & (truth > 0) & np.isfinite(pred) & (pred > 0) & np.isfinite(pyr) & (pyr > 0)
        T = _smooth(_zmask(truth, m)); F = _smooth(_zmask(pred, m)); P = _smooth(_zmask(pyr, m))

        # pericentromere = the single contiguous low-recombination block around the central
        # minimum of the (heavily smoothed) true map -- one clean band, not scattered windows.
        lo, hi = _pericentromere(cen, _zmask(truth, m), m)
        if lo is not None:
            ax.axvspan(cen[lo], cen[hi], color="#eceff1", lw=0, zorder=0)

        ax.axhline(0, color="0.8", lw=0.6, zorder=1)
        ax.plot(cen, T, color=ps.C["truth"], lw=1.9, zorder=4)
        ax.plot(cen, F, color=ps.C["fastrho"], lw=1.5, zorder=3)
        ax.plot(cen, P, color=ps.C["pyrho"], lw=1.4, alpha=0.9, zorder=2)

        rf = float(d[f"c{c}_r"]); rp = float(d[f"c{c}_pyrho_r"])
        ax.text(0.5, 1.13, f"chr {c}", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10.5)
        bbox = dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.6)
        ax.text(0.035, 0.975, f"fastrho  $r={rf:+.2f}$", transform=ax.transAxes, ha="left",
                va="top", fontsize=7.8, color=ps.C["fastrho"], fontweight="bold", bbox=bbox)
        ax.text(0.035, 0.875, f"pyrho  $r={rp:+.2f}$", transform=ax.transAxes, ha="left",
                va="top", fontsize=7.8, color=ps.C["pyrho"], fontweight="bold", bbox=bbox)
        # exemplar mechanism annotation on chr1: pyrho's false hotspot sits in the cold pericentromere
        if c == "1" and lo is not None:
            seg = np.arange(lo, hi + 1)
            ipk = seg[np.nanargmax(P[seg])]
            ax.annotate("pyrho: false hotspot\nin the cold pericentromere",
                        xy=(cen[ipk], min(P[ipk], 4.2)), xytext=(cen[ipk] + 1.6, 4.25),
                        fontsize=6.8, color=ps.C["pyrho"], ha="left", va="top",
                        annotation_clip=True,
                        arrowprops=dict(arrowstyle="->", color=ps.C["pyrho"], lw=1.0))
        ax.set_ylim(-2.8, 4.5)
        ax.set_xlabel("position (Mb)", fontsize=8.6)
        if j == 0:
            ax.set_ylabel("standardized rate ($z$)")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=8.2)
    # one shared legend + the pericentromere note, on the first axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([0], [0], color=ps.C["truth"], lw=1.9, label="true map (Salomé/TAIR10)"),
               Line2D([0], [0], color=ps.C["fastrho"], lw=1.9, label="fastrho (selfing-aware)"),
               Line2D([0], [0], color=ps.C["pyrho"], lw=1.9, label="pyrho"),
               Patch(facecolor="#eceff1", edgecolor="none",
                     label="pericentromere (cold-map block)")]
    axes[0].figure.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.8,
                          bbox_to_anchor=(0.5, 1.005), handlelength=1.6, columnspacing=1.4)
    ps.panel(axes[0], "a", x=-0.42, y=1.30)


def panel_signflip(ax, d):
    from matplotlib.transforms import blended_transform_factory
    x = np.arange(len(CHROMS))
    rf = np.array([float(d[f"c{c}_r"]) for c in CHROMS])
    rp = np.array([float(d[f"c{c}_pyrho_r"]) for c in CHROMS])
    M = max(0.55, 1.2 * float(np.max(np.abs(np.r_[rf, rp]))))  # symmetric, always shows the dots
    edge = blended_transform_factory(ax.transAxes, ax.transData)  # x in axes frac, y in data

    ax.axhspan(-M, 0, color=ps.C["pyrho"], alpha=0.06, lw=0, zorder=0)
    ax.axhspan(0, M, color=ps.C["fastrho"], alpha=0.06, lw=0, zorder=0)
    ax.axhline(0, color="#555", lw=1.0, zorder=1)

    # connector per chromosome: the sign flip is a vertical jump across zero
    for xi, a, b in zip(x, rp, rf):
        ax.plot([xi, xi], [a, b], color="#bbb", lw=1.3, zorder=2)
    ax.scatter(x, rf, s=95, color=ps.C["fastrho"], edgecolor="k", linewidth=0.7, zorder=4,
               label="fastrho (selfing-aware)")
    ax.scatter(x, rp, s=95, color=ps.C["pyrho"], edgecolor="k", linewidth=0.7, zorder=4,
               label="pyrho")
    for xi, v in zip(x, rf):
        ax.annotate(f"{v:+.2f}", (xi, v), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.0, color=ps.C["fastrho"])
    for xi, v in zip(x, rp):
        ax.annotate(f"{v:+.2f}", (xi, v), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8.0, color=ps.C["pyrho"])

    ax.axhline(rf.mean(), color=ps.C["fastrho"], ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.axhline(rp.mean(), color=ps.C["pyrho"], ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.text(0.995, rf.mean(), f"fastrho mean $r={rf.mean():+.2f}$", transform=edge,
            fontsize=8.2, color=ps.C["fastrho"], ha="right", va="bottom")
    ax.text(0.995, rp.mean(), f"pyrho mean $r={rp.mean():+.2f}$", transform=edge,
            fontsize=8.2, color=ps.C["pyrho"], ha="right", va="top")
    ax.text(0.012, 0.965, "recovers the map", transform=ax.transAxes, fontsize=8.0,
            color=ps.C["fastrho"], va="top", ha="left", style="italic")
    ax.text(0.012, 0.035, "inverts the map (anti-correlated)", transform=ax.transAxes,
            fontsize=8.0, color=ps.C["pyrho"], va="bottom", ha="left", style="italic")

    ax.set_xticks(x); ax.set_xticklabels([f"chr {c}" for c in CHROMS])
    ax.set_xlim(-0.5, len(CHROMS) - 0.5)
    ax.set_ylim(-M, M)
    ax.set_ylabel("genome-wide Pearson $r$ vs true map")
    ax.set_title("one naive method, inverted on every chromosome of a real selfer genome",
                 fontsize=10.2, loc="left")
    ax.grid(axis="x", visible=False)
    ps.panel(ax, "b", x=-0.075, y=1.08)


def main():
    d = np.load(DATA, allow_pickle=True)
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.92], hspace=0.42)
    panel_trajectories(fig, gs[0], d)
    panel_signflip(fig.add_subplot(gs[1]), d)
    ps.save(fig, "fig_selfer_inversion", outdir=OUT, formats=("pdf",))
    fig.savefig(os.path.join(OUT, "fig_selfer_inversion.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
