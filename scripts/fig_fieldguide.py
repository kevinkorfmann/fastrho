"""Extended Data figure for the field-guide section: one frozen model applied blind to an
understudied taxon (redpoll, Acanthis flammea; no published recombination map), passing the
blind-QC suite and recovering the chr1 supergene inversion as a recombination cold region.

Reads paper/figdata/fieldguide_redpoll.npz (produced by scripts/fieldguide_run.py + add_pca on
sesame). Render:  python3.13 scripts/fig_fieldguide.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

ps.style()
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, "..", "paper", "figdata", "fieldguide_redpoll.npz"), allow_pickle=True)

INV0, INV1 = 18.9, 75.0                                   # inversion extent, Mb (Funk et al. 2021)
CM = 1e8                                                  # r [Morgan/bp] -> cM/Mb
fastrho, light, green = ps.C["fastrho"], ps.C["fastrho_l"], ps.C["pyrho"]
KARY = [fastrho, "#ff7f00", green]                        # arrangement-1 / heterokaryotype / arrangement-2

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
(axA, axB), (axC, axD) = axes

# --- (a) chr1 recombination map with the supergene inversion shaded --------------------
s = z["starts_1m"] / 1e6
r = z["r_1m"] * CM
axA.axvspan(INV0, INV1, color=ps.HIGHLIGHT, zorder=0)
axA.plot(s, r, color=fastrho, lw=2.0)
axA.set_yscale("log")
axA.set_xlabel("position on redpoll chr1 (Mb)")
axA.set_ylabel(r"$\hat r$  (cM/Mb, LD-based)")
axA.set_title("First fine-scale recombination map of the redpoll", fontsize=10.5)
axA.annotate("supergene inversion\n(18.9–75 Mb): recombination\nsuppressed 2.2×",
             xy=((INV0 + INV1) / 2, np.nanmedian(r) * 3), ha="center", va="center",
             fontsize=8.6, color="#7a5b2b")
ps.panel(axA, "a")

# --- (b) subsample reproducibility (blind precision check) -----------------------------
a, b = z["repro_half1"], z["repro_half2"]
ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
a, b = a[ok] * CM, b[ok] * CM
axB.scatter(a, b, s=7, color=fastrho, alpha=0.35, edgecolors="none")
lim = [min(a.min(), b.min()), max(a.max(), b.max())]
axB.plot(lim, lim, color="#888", lw=1.0, ls="--")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel(r"$\hat r$, sample half 1 (cM/Mb)")
axB.set_ylabel(r"$\hat r$, sample half 2 (cM/Mb)")
axB.set_title("Subsample reproducibility", fontsize=10.5)
axB.text(0.05, 0.93, f"log-Pearson $r={float(z['repro']):.2f}$\n(disjoint halves, $n{{=}}36$ each)",
         transform=axB.transAxes, va="top", fontsize=9, color="#333")
ps.panel(axB, "b")

# --- (c) karyotype PCA over the inversion: the arrangement segregates (37/7/28) ---------
pc1, pc2, lab = z["pca_pc1"], z["pca_pc2"], z["pca_labels"]
sizes = list(z["kary_sizes"])
names = [f"arrangement A ({sizes[0]})", f"heterokaryotype ({sizes[1]})", f"arrangement B ({sizes[2]})"]
for k in range(3):
    m = lab == k
    axC.scatter(pc1[m], pc2[m], s=34, color=KARY[k], edgecolors="white", linewidths=0.5, label=names[k])
axC.set_xlabel(f"PC1 ({z['pca_ev'][0]*100:.0f}% of variance)")
axC.set_ylabel(f"PC2 ({z['pca_ev'][1]*100:.0f}%)")
axC.set_title("Inversion karyotypes segregate (chr1 PCA)", fontsize=10.5)
axC.legend(loc="upper center", fontsize=7.8, handletextpad=0.2, borderpad=0.2)
ps.panel(axC, "c")

# --- (d) inversion suppression: per-interval rate inside vs collinear flanks ------------
pl = z["pos_left"]; rho = z["rho_per_bp"]
inv = (pl >= INV0 * 1e6) & (pl < INV1 * 1e6)
flank = (pl < INV0 * 1e6) | (pl >= INV1 * 1e6)
din = rho[inv]; dfl = rho[flank]
din = din[np.isfinite(din) & (din > 0)]; dfl = dfl[np.isfinite(dfl) & (dfl > 0)]
parts = axD.violinplot([np.log10(dfl), np.log10(din)], showmedians=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor([light, fastrho][i]); pc.set_alpha(0.7); pc.set_edgecolor(fastrho)
parts["cmedians"].set_color("#222")
axD.set_xticks([1, 2]); axD.set_xticklabels(["collinear\nflanks", "inside\ninversion"])
axD.set_ylabel(r"$\log_{10}\ \hat\rho$  (per-interval)")
axD.set_title("Recombination suppressed across the inversion", fontsize=10.5)
axD.text(0.5, 0.06, f"inside/flank $={float(z['inv_ratio']):.2f}$   Mann–Whitney $p<10^{{-300}}$",
         transform=axD.transAxes, ha="center", fontsize=8.6, color="#333")
ps.panel(axD, "d")

fig.tight_layout(w_pad=2.4, h_pad=2.6)
ps.save(fig, "fig_fieldguide")
