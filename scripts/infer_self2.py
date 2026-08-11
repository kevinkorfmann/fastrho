"""Run the improved selfing model (self2) on a real selfer npz and validate vs its stdpopsim map.
Usage (sesame /home/kkor/fastrho): PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 python scripts/infer_self2.py <key>"""
import os
import sys
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastrho.translate import load_model, predict_map_from_genotype_matrix
from fastrho.preprocess import mean_rate_between
import realdata_infer as RI

import json
key = sys.argv[1]
out_json = sys.argv[2] if len(sys.argv) > 2 else ""
ck = open("/home/kkor/fastrho_data/campaign_self2/ckpt.txt").read().strip()
stats = "/home/kkor/fastrho_data/campaign_self2/shards/feat_stats.npz"
z = np.load(f"/home/kkor/realdata/hap/{key}.npz", allow_pickle=True)
gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
chrom = str(z["chrom"]); map_sp = str(z["map_sp"]); map_id = str(z["map_id"])
model, cfg, st = load_model(ck, stats, device="cuda:0")
pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, st, mutation_rate=mu, Ne=None, device="cuda:0")
bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
lo, hi = int(bp[0]), int(bp[-1]); W = 100000
edges = np.append(np.arange(lo, hi, W), hi)
pr = mean_rate_between(bp, pred["r_per_bp"], edges)
tr = RI.truth_windows(map_sp, map_id, chrom, edges)
k = min(len(pr), len(tr)); p, t = pr[:k], tr[:k]
ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
r = float(pearsonr(p[ok], t[ok])[0])
print(f"SELF2 {key}: Pearson={r:.3f} Spearman={spearmanr(p[ok],t[ok])[0]:.3f} "
      f"windows={ok.sum()} vs {map_id}")
if out_json:
    out = dict(n_hap=int(gm.shape[0]), n_snp_used=int(gm.shape[1]), map_id=map_id, model="selfing",
               pearson_vs_map=round(r, 3), windows=int(ok.sum()),
               track=dict(centers=((edges[:-1] + W / 2)[:k][ok] / 1e6).round(3).tolist(),
                          pred=p[ok].tolist(), truth=t[ok].tolist()))
    json.dump(out, open(out_json, "w"))
    print("wrote", out_json)
