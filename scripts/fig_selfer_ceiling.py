"""Extended Data figure: the selfer recovery is CEILING-LIMITED, not method-limited.

The story, in numbers (paper/figdata/selfer_ceiling.json + selfer_chroms.npz + rowan_map.npz):
  (a) A recovery ladder. Clean-sim recovery under the exact map is 0.88 (the method + selfing can
      recover a selfer map that well when the reference is perfect). Two INDEPENDENT published
      A. thaliana maps (Salome vs Rowan) agree only at ~0.55 at 100 kb -- the truth-map ceiling: no
      method can score higher against Salome. fastrho reaches ~half that ceiling (0.27 vs Salome,
      0.32 vs the higher-resolution Rowan map), while pyrho is anti-correlated (-0.22).
  (b) Per chromosome, fastrho is positive against BOTH truth maps and recovers Rowan at least as
      well as Salome (it tracks the true landscape better than the coarse Salome map represents it);
      pyrho inverts against both. The win is robustness, not high absolute fidelity -- and the
      absolute number is bounded by the reference map itself.

Run: PYTHONNOUSERSITE=1 python scripts/fig_selfer_ceiling.py
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDATA = os.path.join(_HERE, "paper", "figdata")
OUT = os.path.join(_HERE, "paper", "figures")
CHROMS = ["1", "2", "3", "4", "5"]

ps.style()
C = ps.C


def panel_ladder(ax, d):
    """Horizontal recovery ladder from clean-sim ceiling down through truth-ceiling to the methods."""
    rows = [
        ("clean-sim recovery\n(exact map, method ceiling)", d["clean_sim_recovery"]["selfer_self2"], "0.55"),
        ("truth-map ceiling\n(Salomé ↔ Rowan, 100 kb)", d["truth_map_ceiling_salome_vs_rowan"]["selfer_windows_100kb"], C["truth"]),
        ("fastrho vs Rowan", d["fastrho_real_recovery"]["vs_rowan_mean"], C["fastrho"]),
        ("fastrho vs Salomé", d["fastrho_real_recovery"]["vs_salome_mean"], C["fastrho_l"]),
        ("pyrho vs Salomé", d["pyrho_real_recovery"]["vs_salome_mean"], C["pyrho"]),
    ]
    y = np.arange(len(rows))[::-1]
    ax.axvline(0, color="#888", lw=1.0, zorder=1)
    ax.axvline(0.55, color=C["truth"], ls="--", lw=1.0, alpha=0.6, zorder=1)
    for yi, (lab, val, col) in zip(y, rows):
        c = col if col.startswith("#") else "#7f7f7f"
        ax.plot([0, val], [yi, yi], color=c, lw=2.4, zorder=2)
        ax.scatter([val], [yi], s=70, color=c, edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(f"{val:+.2f}", (val, yi), xytext=(8 if val >= 0 else -8, 0),
                    textcoords="offset points", va="center", ha="left" if val >= 0 else "right",
                    fontsize=9, color=c if c != "#7f7f7f" else "#333")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.6)
    ax.set_xlim(-0.35, 1.0); ax.set_xlabel("100 kb map-recovery Pearson $r$")
    ax.text(0.55, len(rows) - 0.4, "truth-map ceiling", color=C["truth"], fontsize=8.2,
            ha="center", va="bottom", style="italic")
    ax.text(-0.33, 0.15, "inverted", color=C["pyrho"], fontsize=8.0, ha="left", style="italic")
    ax.set_title("(a) recovery is bounded by the truth map, not the method", loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)


def panel_perchrom(ax, d):
    fr_s = np.array(d["fastrho_real_recovery"]["per_chrom_vs_salome"])
    fr_r = np.array(d["fastrho_real_recovery"]["per_chrom_vs_rowan"])
    py_s = np.array(d["pyrho_real_recovery"]["per_chrom_vs_salome"])
    x = np.arange(len(CHROMS)); w = 0.26
    ax.axhline(0, color="#888", lw=1.0)
    ax.bar(x - w, fr_s, w, color=C["fastrho_l"], label="fastrho vs Salomé", edgecolor="k", linewidth=0.4)
    ax.bar(x,     fr_r, w, color=C["fastrho"],   label="fastrho vs Rowan",  edgecolor="k", linewidth=0.4)
    ax.bar(x + w, py_s, w, color=C["pyrho"],     label="pyrho vs Salomé",   edgecolor="k", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([f"chr {c}" for c in CHROMS])
    ax.set_ylabel("100 kb Pearson $r$ vs true map"); ax.set_ylim(-0.45, 0.55)
    ax.legend(frameon=False, fontsize=8.0, ncol=1, loc="lower left")
    ax.set_title("(b) fastrho positive on every chromosome vs both maps; pyrho inverts", loc="left", fontsize=10)
    ax.grid(axis="x", visible=False)


def panel_decomp(ax, d):
    """Recovery under each real-data confound (vs the EXACT map): none reaches down to the truth-map
    ceiling, so the real-data gap is truth-map noise, not a fixable modelling confound."""
    g = d["gap_decomposition"]
    rows = [("clean sim", g["clean_sim"]),
            ("+ structure (Fst 0.2)", g["structure_split_Fst0.2"]),
            ("+ genotyping error", g["genotyping_error_0.2pct"]),
            ("+ missing→ref (10%)", g["missing_to_ref_10pct"]),
            ("+ polarization flip", g["polarization_flip_10pct"]),
            ("+ demographic expansion", g["demographic_expansion_SMA"])]
    y = np.arange(len(rows))[::-1]
    ceil = d["truth_map_ceiling_salome_vs_rowan"]["selfer_windows_100kb"]
    real = d["fastrho_real_recovery"]["vs_salome_mean"]
    ax.barh(
        y,
        [r[1] for r in rows],
        color="white",
        edgecolor=C["fastrho"],
        linewidth=1.0,
        height=0.58,
        zorder=2,
    )
    ax.axvline(ceil, color=C["truth"], ls="--", lw=1.2, zorder=3)
    ax.axvline(real, color=C["fastrho"], ls=":", lw=1.2, zorder=3)
    ax.text(ceil, len(rows) - 0.3, "truth-map\nceiling %.2f" % ceil, color=C["truth"], fontsize=7.6, ha="center", va="bottom")
    ax.text(real, -0.9, "real fastrho\n%.2f" % real, color=C["fastrho"], fontsize=7.6, ha="center", va="top")
    for yi, (lab, val) in zip(y, rows):
        ax.annotate(f"{val:.2f}", (val, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.4)
    ax.set_xlim(0, 1.0); ax.set_xlabel("recovery vs the exact simulated map")
    ax.set_title("(c) no sim confound reaches the truth-map ceiling — the real gap is truth-map noise",
                 loc="left", fontsize=10)
    ax.grid(axis="y", visible=False)


def main():
    d = json.loads(open(os.path.join(FIGDATA, "selfer_ceiling.json")).read())
    from matplotlib import gridspec
    fig = plt.figure(figsize=(12.2, 8.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.42, height_ratios=[1.0, 0.85],
                           width_ratios=[1.15, 1.0])
    axa = fig.add_subplot(gs[0, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, :])
    panel_ladder(axa, d); panel_perchrom(axb, d); panel_decomp(axc, d)
    ps.panel(axa, "a", x=-0.62, y=1.06); ps.panel(axb, "b", x=-0.14, y=1.06); ps.panel(axc, "c", x=-0.30, y=1.06)
    ps.save(fig, "fig_selfer_ceiling", outdir=OUT, formats=("pdf",))
    fig.savefig(os.path.join(OUT, "fig_selfer_ceiling.png"), dpi=200, bbox_inches="tight")
    print("wrote fig_selfer_ceiling")


if __name__ == "__main__":
    main()
