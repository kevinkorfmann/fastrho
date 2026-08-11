"""Turn a published recombination/genetic map (any of the common formats) into a standardized
truth-map npz (fields: pos[bp], rate[per-bp, arbitrary scale], chrom) that scripts/transect_infer.py
validates against via TRUTHMAP_NPZ. Only the SHAPE matters (fastrho scores rank/Pearson correlation),
so unnormalized densities are fine.

Formats (generalizing the crossover->rate idiom of scripts/build_rowan_map.py):
  crossover  file of crossover/breakpoint positions -> CO density per window (col: --pos-col)
  marey      genetic map, marker rows (bp, cM) -> local slope dcM/dbp between consecutive markers
  rmap       LDhat/pyrho-style (pos, rate_per_bp) two/three-column -> direct
  windows    precomputed per-window (start,end,rate) table -> direct

Usage:
  python scripts/build_species_maps.py --format crossover --in FileS2.csv --chrom 1 \
      --pos-col breakpoint.pos --chrom-col chr --win 100000 --out map_<sp>_chr1.npz
  python scripts/build_species_maps.py --format marey --in map.tsv --chrom 3 \
      --bp-col phys --cm-col gen --out map_<sp>_chr3.npz
"""
import os
import csv
import argparse

import numpy as np


def _rows(path):
    # sniff delimiter (csv/tsv), yield dict rows
    with open(path) as fh:
        head = fh.readline()
        delim = "\t" if head.count("\t") >= head.count(",") else ","
    with open(path) as fh:
        yield from csv.DictReader(fh, delimiter=delim)


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def from_crossover(path, chrom, chrom_col, pos_col, win):
    bps = []
    for row in _rows(path):
        if str(row.get(chrom_col, "")).strip().replace("chr", "") == str(chrom).replace("chr", ""):
            v = _num(row.get(pos_col))
            if np.isfinite(v):
                bps.append(v)
    bps = np.sort(np.asarray(bps, float))
    if len(bps) == 0:
        raise SystemExit(f"no crossovers for chrom {chrom} (cols {chrom_col}/{pos_col})")
    edges = np.append(np.arange(bps.min(), bps.max(), win), bps.max())
    counts, _ = np.histogram(bps, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2
    rate = counts / np.maximum(np.diff(edges), 1.0)   # CO density ~ rate
    return centers, rate, f"{len(bps)} crossovers"


def from_marey(path, chrom, chrom_col, bp_col, cm_col):
    P, C = [], []
    for row in _rows(path):
        if chrom_col and str(row.get(chrom_col, "")).strip().replace("chr", "") != str(chrom).replace("chr", ""):
            continue
        bp, cm = _num(row.get(bp_col)), _num(row.get(cm_col))
        if np.isfinite(bp) and np.isfinite(cm):
            P.append(bp); C.append(cm)
    P, C = np.asarray(P, float), np.asarray(C, float)
    o = np.argsort(P); P, C = P[o], C[o]
    # local slope cM per bp -> per-bp rate (cM/bp * 1e-2 = Morgan/bp ~ r); midpoints of intervals
    dcm = np.diff(C); dbp = np.diff(P)
    good = dbp > 0
    rate = np.clip(dcm[good], 0, None) / dbp[good] * 1e-2
    centers = ((P[:-1] + P[1:]) / 2)[good]
    return centers, rate, f"{len(P)} markers"


def from_rmap(path, pos_col=None, rate_col=None):
    P, R = [], []
    rows = list(_rows(path))
    if not rows:  # headerless numeric
        arr = np.loadtxt(path)
        return arr[:, 0].astype(float), arr[:, -1].astype(float), f"{len(arr)} rows (headerless)"
    fields = rows[0].keys()
    pc = pos_col or next(f for f in fields if "pos" in f.lower() or "start" in f.lower())
    rc = rate_col or next(f for f in fields if "rate" in f.lower() or "rho" in f.lower() or "cm" in f.lower())
    for row in rows:
        p, r = _num(row.get(pc)), _num(row.get(rc))
        if np.isfinite(p) and np.isfinite(r):
            P.append(p); R.append(r)
    return np.asarray(P, float), np.asarray(R, float), f"{len(P)} rmap rows"


def from_windows(path, chrom, chrom_col, start_col, end_col, rate_col):
    P, R = [], []
    for row in _rows(path):
        if chrom_col and str(row.get(chrom_col, "")).strip().replace("chr", "") != str(chrom).replace("chr", ""):
            continue
        s, e, r = _num(row.get(start_col)), _num(row.get(end_col)), _num(row.get(rate_col))
        if np.isfinite(s) and np.isfinite(r):
            P.append((s + (e if np.isfinite(e) else s)) / 2); R.append(r)
    return np.asarray(P, float), np.asarray(R, float), f"{len(P)} windows"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["crossover", "marey", "rmap", "windows"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--chrom", default="")
    ap.add_argument("--chrom-col", default="chr")
    ap.add_argument("--pos-col", default="pos")
    ap.add_argument("--bp-col", default="phys")
    ap.add_argument("--cm-col", default="cM")
    ap.add_argument("--start-col", default="start")
    ap.add_argument("--end-col", default="end")
    ap.add_argument("--rate-col", default="rate")
    ap.add_argument("--win", type=int, default=100000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.format == "crossover":
        pos, rate, note = from_crossover(a.inp, a.chrom, a.chrom_col, a.pos_col, a.win)
    elif a.format == "marey":
        pos, rate, note = from_marey(a.inp, a.chrom, a.chrom_col, a.bp_col, a.cm_col)
    elif a.format == "rmap":
        pos, rate, note = from_rmap(a.inp, a.pos_col if a.pos_col != "pos" else None,
                                    a.rate_col if a.rate_col != "rate" else None)
    else:
        pos, rate, note = from_windows(a.inp, a.chrom, a.chrom_col, a.start_col, a.end_col, a.rate_col)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez(a.out, pos=pos.astype(np.float64), rate=rate.astype(np.float64), chrom=str(a.chrom),
             source=os.path.basename(a.inp), fmt=a.format)
    print(f"[map] {a.format} chrom={a.chrom}: {note} -> {a.out} "
          f"({len(pos)} points, rate range {np.nanmin(rate):.2e}..{np.nanmax(rate):.2e})")


if __name__ == "__main__":
    main()
