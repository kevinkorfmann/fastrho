"""Raw LD-decay check for the redpoll inversion arrangements.

This is deliberately independent of fastrho inference.  It reconstructs diploid
dosages from the pseudo-haplotype matrix, samples SNP pairs by physical-distance
bin, and compares finite-sample-corrected dosage r^2 inside the inversion and in
the collinear flanks for the pooled panel and the two homokaryotypes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


BINS = np.array([500, 1_000, 2_000, 5_000, 10_000, 25_000, 50_000,
                 100_000, 250_000, 500_000], dtype=float)


def dosage(gm):
    return (gm[0::2].astype(np.float32) + gm[1::2].astype(np.float32))


def sample_pairs(pos, eligible, lo, hi, n_pairs, rng):
    anchors = np.flatnonzero(eligible)
    out_i, out_j = [], []
    attempts = 0
    while sum(len(x) for x in out_i) < n_pairs and attempts < 30:
        attempts += 1
        m = min(max(4 * (n_pairs - sum(len(x) for x in out_i)), 2000), 200_000)
        ii = rng.choice(anchors, size=m, replace=True)
        target = pos[ii] + rng.uniform(lo, hi, size=m)
        jj = np.searchsorted(pos, target)
        ok = jj < len(pos)
        ii, jj = ii[ok], jj[ok]
        dist = pos[jj] - pos[ii]
        ok = eligible[jj] & (dist >= lo) & (dist < hi)
        out_i.append(ii[ok]); out_j.append(jj[ok])
    if not out_i:
        return np.array([], int), np.array([], int)
    return np.concatenate(out_i)[:n_pairs], np.concatenate(out_j)[:n_pairs]


def ld_curve(D, pos, region, n_pairs, seed):
    n = D.shape[0]
    p = D.mean(axis=0) / 2
    polymorphic = (p >= 0.05) & (p <= 0.95)
    eligible = region & polymorphic
    mean = D.mean(axis=0)
    sd = D.std(axis=0)
    eligible &= sd > 0
    rng = np.random.default_rng(seed)
    rows = []
    for k, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        i, j = sample_pairs(pos, eligible, lo, hi, n_pairs, rng)
        xi = (D[:, i] - mean[i]) / sd[i]
        xj = (D[:, j] - mean[j]) / sd[j]
        r = np.mean(xi * xj, axis=0)
        r2 = np.square(r)
        raw = float(np.mean(r2))
        corrected = max(raw - 1.0 / n, 0.0)
        rows.append({
            "distance_lo": int(lo),
            "distance_hi": int(hi),
            "distance_mid": float(np.sqrt(lo * hi)),
            "n_pairs": int(len(r2)),
            "mean_r2_raw": raw,
            "mean_r2_corrected": corrected,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", required=True)
    ap.add_argument("--fieldguide", required=True)
    ap.add_argument("--pairs", type=int, default=20_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = np.load(args.hap, allow_pickle=True)
    fg = np.load(args.fieldguide, allow_pickle=True)
    D = dosage(raw["gm"])
    pos = raw["pos"].astype(float)
    labels = fg["pca_labels"].astype(int)
    inv_start, inv_end = int(fg["inv_start"]), int(fg["inv_end"])
    sizes = np.bincount(labels, minlength=3)
    homo_labels = [int(x) for x in np.argsort(sizes)[-2:]]
    homo_labels.sort(key=lambda x: float(np.mean(fg["pca_pc1"][labels == x])))
    groups = {
        "pooled": np.arange(len(labels)),
        "arrangement_A": np.flatnonzero(labels == homo_labels[0]),
        "arrangement_B": np.flatnonzero(labels == homo_labels[1]),
    }
    regions = {
        "inside": (pos >= inv_start) & (pos < inv_end),
        "flanks": (pos < inv_start) | (pos >= inv_end),
    }

    out = {
        "bins": BINS.astype(int).tolist(),
        "pairs_per_bin": args.pairs,
        "inv_start": inv_start,
        "inv_end": inv_end,
        "groups": {},
    }
    for gi, (name, inds) in enumerate(groups.items()):
        out["groups"][name] = {"n_individuals": int(len(inds))}
        for ri, (region_name, mask) in enumerate(regions.items()):
            rows = ld_curve(D[inds], pos, mask, args.pairs, 91000 + 100 * gi + ri)
            out["groups"][name][region_name] = rows
            tail = rows[-1]["mean_r2_corrected"]
            print(name, region_name, "corrected r2 at 250-500 kb", f"{tail:.5f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
