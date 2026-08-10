"""Blind subsample-reproducibility QC for transect species with no published map.

The field-guide's key no-ground-truth check: split the sample into two disjoint halves, infer a map from
each with the frozen DR model, and correlate the two recovered maps at 100 kb. High reproducibility ->
the data determine the map (trustworthy even without a truth map); low -> data-limited (flag/drop).

Splits by INDIVIDUAL: the extractor lays out 2 pseudo-haplotype rows per individual (dosage/haploid) or
2 real haplotypes (phased2), so rows come in pairs; we split whole pairs to keep each half's diploids
intact. Uses the same frozen DR unphased engine as transect_infer.

Usage (sesame /home/kkor/fastrho_dr): PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 DEV=cuda:0 \
    python scripts/transect_qc.py <hap.npz> [out.json]
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transect_infer import infer_unphased
from fastrho.preprocess import mean_rate_between

W = 100_000


def main():
    z = np.load(sys.argv[1], allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
    n = gm.shape[0]
    npair = n // 2
    # pair p occupies rows [2p, 2p+1]; assign alternating pairs to halves A/B (disjoint individuals)
    a_pairs = np.arange(0, npair, 2)
    b_pairs = np.arange(1, npair, 2)
    ra = np.concatenate([[2 * p, 2 * p + 1] for p in a_pairs])
    rb = np.concatenate([[2 * p, 2 * p + 1] for p in b_pairs])
    gmA, gmB = gm[ra], gm[rb]

    bpA, rA, nA = infer_unphased(gmA, pos, mu)
    bpB, rB, nB = infer_unphased(gmB, pos, mu)
    lo = int(max(bpA[0], bpB[0])); hi = int(min(bpA[-1], bpB[-1]))
    edges = np.append(np.arange(lo, hi, W), hi)
    mA = mean_rate_between(bpA, rA, edges)
    mB = mean_rate_between(bpB, rB, edges)
    k = min(len(mA), len(mB)); a, b = mA[:k], mB[:k]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    from scipy.stats import pearsonr
    rep = float(pearsonr(a[ok], b[ok])[0]) if ok.sum() > 3 else None
    lrep = float(pearsonr(np.log(a[ok]), np.log(b[ok]))[0]) if ok.sum() > 3 else None
    out = dict(key=os.path.basename(sys.argv[1]).replace(".npz", ""),
               n_hap=int(n), half_n_hap=int(gmA.shape[0]),
               reproducibility=round(rep, 3) if rep is not None else None,
               log_reproducibility=round(lrep, 3) if lrep is not None else None,
               windows=int(ok.sum()))
    print("QC " + json.dumps(out))
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
