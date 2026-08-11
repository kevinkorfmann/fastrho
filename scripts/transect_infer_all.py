"""Batch inference for the tree-of-life transect: load the frozen DR model ONCE, infer every hap npz.

Loops over <hapdir>/*.npz, and for each (skipping any with an existing transect_<key>.json) recovers the
map from unphased genotypes and validates vs stdpopsim (map_sp/map_id in the npz) or a custom map, else
records the novel landscape. Reuses transect_infer.infer_unphased (the model is cached in-process, so 64
species cost one model load). Run from /home/kkor/fastrho_dr.

Usage (sesame): PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 DEV=cuda:0 \
    python scripts/transect_infer_all.py /home/kkor/realdata/hap /home/kkor/realdata
"""
import os
import sys
import glob
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transect_infer import infer_unphased, _custom_truth
from fastrho.preprocess import mean_rate_between
import realdata_infer as RI
from scipy.stats import pearsonr

W = 100_000


def process(npz_path, out_json):
    z = np.load(npz_path, allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
    if gm.shape[0] < 6 or gm.shape[1] < 200:
        return {"key": os.path.basename(npz_path)[:-4], "error": f"too small {gm.shape}"}
    bp, r, nsnp = infer_unphased(gm, pos, mu)
    out = {"n_hap": int(gm.shape[0]), "n_snp_used": int(nsnp)}
    chrom = str(z["chrom"]) if "chrom" in z.files else "?"
    tmap = str(z["truthmap"]) if "truthmap" in z.files else os.environ.get("TRUTHMAP_NPZ", "")
    have_std = "map_sp" in z.files
    lo, hi = int(bp[0]), int(bp[-1])
    edges = np.append(np.arange(lo, hi, W), hi)
    pr = mean_rate_between(bp, r, edges)
    if have_std or (tmap and os.path.exists(tmap)):
        if have_std:
            tr = RI.truth_windows(str(z["map_sp"]), str(z["map_id"]), chrom, edges)
            out["map_id"] = str(z["map_id"])
        else:
            tr = _custom_truth(tmap, chrom, edges); out["map_id"] = os.path.basename(tmap)
        k = min(len(pr), len(tr)); p, t = pr[:k], tr[:k]
        ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
        out["pearson_vs_map"] = round(float(pearsonr(p[ok], t[ok])[0]), 3) if ok.sum() > 3 else None
        out["windows"] = int(ok.sum())
        out["track"] = dict(centers=((edges[:-1] + W / 2)[:k][ok] / 1e6).round(3).tolist(),
                            pred=p[ok].tolist(), truth=t[ok].tolist())
    else:
        fin = np.isfinite(pr) & (pr > 0)
        out["windows"] = int(fin.sum())
        out["track"] = dict(centers=((edges[:-1] + W / 2)[fin] / 1e6).round(3).tolist(),
                            pred=pr[fin].tolist(), truth=None)
    json.dump(out, open(out_json, "w"))
    return {"key": os.path.basename(npz_path)[:-4], "n_hap": out["n_hap"], "n_snp": out["n_snp_used"],
            "windows": out["windows"], "r": out.get("pearson_vs_map")}


def main():
    hapdir = sys.argv[1]; outdir = sys.argv[2]
    only = set(sys.argv[3:])  # optional: restrict to these keys
    limit = int(os.environ.get("LIMIT", "0"))  # max species to process this call (0 = all)
    done = 0
    for npz in sorted(glob.glob(os.path.join(hapdir, "*.npz"))):
        key = os.path.basename(npz)[:-4]
        if only and key not in only:
            continue
        outj = os.path.join(outdir, f"transect_{key}.json")
        if os.path.exists(outj):
            continue
        try:
            res = process(npz, outj)
            print("INFER " + json.dumps(res))
        except Exception as e:
            print(f"INFER_ERR {key}: {type(e).__name__}: {str(e)[:120]}")
        done += 1
        if limit and done >= limit:
            break


if __name__ == "__main__":
    main()
