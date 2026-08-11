"""Featurize one real A. thaliana chromosome ONCE and cache the raw SNP-token features.

The LD featurization of a full chromosome (~600k-850k SNPs, 6 radii) is CPU-bound and takes
~8 min. It is MODEL-INDEPENDENT, so for a 50-epoch real-data checkpoint sweep it must be done
once per chromosome, not once per checkpoint. This caches raw (unstandardized) tokens; each
checkpoint then standardizes with its own feat_stats and runs a cheap GPU forward via
predict_from_tokens (see self_epoch_select.py --cache-dir).

Pipeline is byte-identical to realdata_infer.py's athal path: basic_filtering -> SNPTokenFeaturizer()
with default config (that is what translate.predict_intervals uses for the non-gt fastrho models).

Usage: python scripts/self_featurize_cache.py <chrom 1..5> <out_dir>   # run 5 in parallel
"""
import sys, os
import numpy as np
from fastcxt.sfs import basic_filtering
from fastrho.features import SNPTokenFeaturizer

HAP = "/home/kkor/realdata/hap"


def main():
    chrom = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"athal_c{chrom}_tokens.npz")
    if os.path.exists(out):
        print(f"chr{chrom}: cache exists, skip -> {out}"); return
    z = np.load(f"{HAP}/athal_c{chrom}.npz", allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64)
    mu = float(z["mu"])
    gm, pos = basic_filtering(gm.astype(np.int8), pos)
    feats = SNPTokenFeaturizer()(gm, pos, {"sequence_length": float(pos[-1] + 1)})
    np.savez(out, tokens=np.asarray(feats["tokens"], np.float32), positions=pos,
             n_hap=int(gm.shape[0]), mu=mu, chrom=str(z["chrom"]),
             map_sp=str(z["map_sp"]), map_id=str(z["map_id"]))
    print(f"chr{chrom}: cached {feats['tokens'].shape[0]} SNP-tokens x {feats['tokens'].shape[1]} "
          f"feat, {gm.shape[0]} hap -> {out}")


if __name__ == "__main__":
    main()
