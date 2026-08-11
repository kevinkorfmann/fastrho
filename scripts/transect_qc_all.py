"""Batch blind-QC for the transect: load the frozen DR model once, run subsample reproducibility on
every hap npz that has no published map (writes qc_<key>.json). Reuses transect_qc's split logic.

Usage (sesame /home/kkor/fastrho_dr): PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 DEV=cuda:0 \
    python scripts/transect_qc_all.py /home/kkor/realdata/hap /home/kkor/realdata
"""
import os
import sys
import glob
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transect_infer import infer_unphased
from fastrho.preprocess import mean_rate_between
from scipy.stats import pearsonr

W = 100_000


def qc(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
    n = gm.shape[0]; npair = n // 2
    if npair < 8:
        return None
    a = np.concatenate([[2 * p, 2 * p + 1] for p in range(0, npair, 2)])
    b = np.concatenate([[2 * p, 2 * p + 1] for p in range(1, npair, 2)])
    bpA, rA, _ = infer_unphased(gm[a], pos, mu)
    bpB, rB, _ = infer_unphased(gm[b], pos, mu)
    lo = int(max(bpA[0], bpB[0])); hi = int(min(bpA[-1], bpB[-1]))
    edges = np.append(np.arange(lo, hi, W), hi)
    mA = mean_rate_between(bpA, rA, edges); mB = mean_rate_between(bpB, rB, edges)
    k = min(len(mA), len(mB)); x, y = mA[:k], mB[:k]
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() <= 3:
        return None
    return dict(n_hap=int(n), half_n_hap=int(len(a)),
                reproducibility=round(float(pearsonr(x[ok], y[ok])[0]), 3),
                log_reproducibility=round(float(pearsonr(np.log(x[ok]), np.log(y[ok]))[0]), 3),
                windows=int(ok.sum()))


def main():
    hapdir = sys.argv[1]; outdir = sys.argv[2]
    only = set(sys.argv[3:])
    limit = int(os.environ.get("LIMIT", "0"))
    done = 0
    for npz in sorted(glob.glob(os.path.join(hapdir, "*.npz"))):
        key = os.path.basename(npz)[:-4]
        if only and key not in only:
            continue
        # skip if this species is validated (has a map) or already QC'd
        tj = os.path.join(outdir, f"transect_{key}.json")
        if os.path.exists(tj) and json.load(open(tj)).get("pearson_vs_map") is not None:
            continue
        if os.path.exists(os.path.join(outdir, f"qc_{key}.json")):
            continue
        try:
            r = qc(npz)
            if r:
                json.dump(r, open(os.path.join(outdir, f"qc_{key}.json"), "w"))
                print(f"QC {key}: repro={r['reproducibility']} logrepro={r['log_reproducibility']} "
                      f"n_hap={r['n_hap']} win={r['windows']}")
            else:
                print(f"QC_SKIP {key}: too few pairs/windows")
        except Exception as e:
            print(f"QC_ERR {key}: {type(e).__name__}: {str(e)[:120]}")
        done += 1
        if limit and done >= limit:
            break


if __name__ == "__main__":
    main()
