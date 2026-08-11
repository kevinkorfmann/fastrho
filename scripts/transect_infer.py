"""Frozen DR (domain-randomized) UNPHASED inference — the engine for the tree-of-life transect.

One model, phase- AND polarization-invariant composite-LD features, NO retraining. Given a hap/dosage
npz (gm, pos, mu, and optionally chrom/map_sp/map_id), infer the fine-scale recombination map from
UNPHASED genotypes and, if a validation map is provided, score recovery at 100 kb. pyrho cannot run on
this input at all. Reuses the exact DR path from scripts/stdpopsim_maps.py.

Usage (sesame): CUDA_VISIBLE_DEVICES=1 DEV=cuda:0 python scripts/transect_infer.py <hap.npz> [out.json]
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.translate import load_model, predict_intervals, dr_variant_stats
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig
from fastrho.preprocess import mean_rate_between
from fastcxt.sfs import basic_filtering
import realdata_infer as RI

DR_CKPT = os.environ.get("DR_CKPT",
    "/home/kkor/fastrho_data/campaign/train_dr15k/fastrho/version_0/checkpoints/epoch=53-val_pearson=0.862.ckpt")
DR_STATS = os.environ.get("DR_STATS", "/home/kkor/fastrho_data/campaign/shards_dr15k/feat_stats_dr.npz")
DEV = os.environ.get("DEV", "cuda:0")
W = 100_000

_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        model, mcfg, stats = load_model(DR_CKPT, DR_STATS, device=DEV)
        stats = dr_variant_stats(stats, "gtf")   # phase- + polarization-invariant view
        feat = GTTokenFeaturizer(config=FeatureConfig(sfs_shape=True, r2_debias=True), fold=True)
        _MODEL = (model, mcfg, stats, feat)
    return _MODEL


def infer_unphased(gm, pos, mu, Ne=None):
    """Genotype matrix -> (bp_edges, r_per_bp) via the frozen DR unphased+unpolarized model."""
    model, mcfg, stats, feat = _load()
    gmf, posf = basic_filtering(gm.astype(np.int8), pos.astype(np.float64))
    pred = predict_intervals(model, mcfg, stats, gmf, posf, mu, Ne=Ne, device=DEV,
                             featurizer=feat, cond_extra=[1.0, 1.0])
    return np.r_[pred["pos_left"][0], pred["pos_right"]], pred["r_per_bp"], gmf.shape[1]


def _custom_truth(path, chrom, edges):
    """Load a custom validation map npz (fields: pos[bp], rate[per-bp r]; optional chrom) and window it.
    Used for species not in stdpopsim (most of the transect). See scripts/build_species_maps.py."""
    m = np.load(path, allow_pickle=True)
    if "chrom" in m.files and str(m["chrom"]) not in ("", chrom, chrom.replace("chr", "")):
        pass  # single-contig map files are fine; multi-contig callers pass the matching file
    tp = np.asarray(m["pos"], float); tr = np.asarray(m["rate"], float)
    tr = np.where(np.isfinite(tr), tr, 0.0)
    return mean_rate_between(tp, tr, edges)


def main():
    z = np.load(sys.argv[1], allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
    bp, r, nsnp = infer_unphased(gm, pos, mu)
    out = {"n_hap": int(gm.shape[0]), "n_snp_used": int(nsnp)}
    chrom = str(z["chrom"]) if "chrom" in z.files else "?"
    truthmap = os.environ.get("TRUTHMAP_NPZ", "")   # custom (non-stdpopsim) validation map
    have_std = "map_sp" in z.files
    if have_std or (truthmap and os.path.exists(truthmap)):
        lo, hi = int(bp[0]), int(bp[-1])
        edges = np.append(np.arange(lo, hi, W), hi)
        pr = mean_rate_between(bp, r, edges)
        if have_std:
            tr = RI.truth_windows(str(z["map_sp"]), str(z["map_id"]), chrom, edges)
            out["map_id"] = str(z["map_id"])
        else:
            tr = _custom_truth(truthmap, chrom, edges)
            out["map_id"] = os.path.basename(truthmap)
        k = min(len(pr), len(tr)); p, t = pr[:k], tr[:k]
        ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
        from scipy.stats import pearsonr
        out["pearson_vs_map"] = round(float(pearsonr(p[ok], t[ok])[0]), 3) if ok.sum() > 3 else None
        out["windows"] = int(ok.sum())
        out["track"] = dict(centers=((edges[:-1] + W / 2)[:k][ok] / 1e6).round(3).tolist(),
                            pred=p[ok].tolist(), truth=t[ok].tolist())
    else:
        # novel species (no map): still save the recovered landscape for the figure + blind QC
        lo, hi = int(bp[0]), int(bp[-1])
        edges = np.append(np.arange(lo, hi, W), hi)
        pr = mean_rate_between(bp, r, edges)
        fin = np.isfinite(pr) & (pr > 0)
        out["windows"] = int(fin.sum())
        out["track"] = dict(centers=((edges[:-1] + W / 2)[fin] / 1e6).round(3).tolist(),
                            pred=pr[fin].tolist(), truth=None)
    print("RESULT " + json.dumps({k: v for k, v in out.items() if k != "track"}))
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"))
        print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
