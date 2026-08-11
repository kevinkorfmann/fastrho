"""Build an A. thaliana recombination map from the Rowan et al. 2019 crossover set (17,077 COs /
2,182 F2, Col x Ler, TAIR10-native) and correlate it against the stdpopsim SalomeAveraged_TAIR10
map -- the TRUTH-NOISE ceiling: no method can score a recovered map above the agreement between two
independent "truth" maps.

Rowan FileS2 columns: chr, block1.end.pos, block2.start.pos, breakpoint.pos, Selected.420.
CO density (breakpoints binned per window) is proportional to the recombination rate. We compare the
two maps at 100 kb (our scoring resolution) and 200 kb (Rowan's own window).

Usage: python scripts/build_rowan_map.py <rowan_FileS2.csv> [out_npz]
"""
import os
import sys

import numpy as np
import stdpopsim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.preprocess import mean_rate_between

CHROMS = ["1", "2", "3", "4", "5"]


def load_rowan(csv_path):
    """-> {chrom: sorted breakpoint positions (bp)}."""
    import csv
    by = {c: [] for c in CHROMS}
    with open(csv_path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            c = row["chr"].strip()
            if c in by:
                by[c].append(float(row["breakpoint.pos"]))
    return {c: np.sort(np.asarray(v, float)) for c, v in by.items()}


def rowan_rate_at(breakpoints, edges):
    """CO count per window / window width -> per-bp recombination density (unnormalised rate)."""
    counts, _ = np.histogram(breakpoints, bins=edges)
    width = np.diff(edges)
    return counts / np.maximum(width, 1.0)


def main():
    csv_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figdata", "rowan_map.npz")
    bp = load_rowan(csv_path)
    sp = stdpopsim.get_species("AraTha")
    gm = sp.get_genetic_map("SalomeAveraged_TAIR10")

    save = {}
    print("Salome vs Rowan truth-map agreement (Pearson of per-window rate):")
    print(f"{'chr':>3} {'nCO':>6} {'r@100kb':>9} {'r@200kb':>9} {'sp@100kb':>9}")
    for W, tag in ((100_000, "100kb"), (200_000, "200kb")):
        allp = {"sal": [], "row": []}
        for c in CHROMS:
            rmc = gm.get_chromosome_map(c)
            pos = np.asarray(rmc.position, float); rate = np.where(np.isfinite(rmc.rate), rmc.rate, 0.0)
            L = pos[-1]
            edges = np.append(np.arange(0, L, W), L)
            sal = mean_rate_between(pos, rate, edges)
            row = rowan_rate_at(bp[c], edges)
            ok = np.isfinite(sal) & (sal > 0) & np.isfinite(row) & (row > 0)
            save[f"c{c}_edges_{tag}"] = edges
            save[f"c{c}_salome_{tag}"] = sal
            save[f"c{c}_rowan_{tag}"] = row
            allp["sal"].append(sal[ok]); allp["row"].append(row[ok])
            if W == 100_000:
                from scipy.stats import spearmanr
                r100 = float(np.corrcoef(sal[ok], row[ok])[0, 1])
                sp100 = float(spearmanr(sal[ok], row[ok])[0])
                save[f"c{c}_nco"] = len(bp[c])
                # store for the row print after we also have 200kb
                save.setdefault("_r100", {})[c] = (len(bp[c]), r100, sp100)
        S = np.concatenate(allp["sal"]); R = np.concatenate(allp["row"])
        rg = float(np.corrcoef(S, R)[0, 1])
        save[f"pooled_r_{tag}"] = rg
        if W == 100_000:
            for c in CHROMS:
                nco, r100, sp100 = save["_r100"][c]
                save[f"c{c}_r_100kb"] = r100
                print(f"{c:>3} {nco:>6} {r100:>9.3f} {'':>9} {sp100:>9.3f}")
    # fill 200kb per-chrom
    for c in CHROMS:
        sal = save[f"c{c}_salome_200kb"]; row = save[f"c{c}_rowan_200kb"]
        ok = np.isfinite(sal) & (sal > 0) & np.isfinite(row) & (row > 0)
        save[f"c{c}_r_200kb"] = float(np.corrcoef(sal[ok], row[ok])[0, 1])
    print("-" * 42)
    print(f"POOLED  r@100kb = {save['pooled_r_100kb']:.3f}   r@200kb = {save['pooled_r_200kb']:.3f}")
    print("per-chrom r@200kb:", {c: round(save[f'c{c}_r_200kb'], 3) for c in CHROMS})
    save.pop("_r100", None)
    np.savez(out, **{k: v for k, v in save.items() if not k.startswith("_")})
    print("wrote", out)


if __name__ == "__main__":
    main()
