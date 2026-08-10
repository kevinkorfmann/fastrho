"""Blind GoF via SUBSAMPLE REPRODUCIBILITY (the blind cousin of the Fig-6 noise floor).

A map is trustworthy only if the data DETERMINE it: split the sample in two halves, infer the
map independently on each, and correlate the two half-maps. High reproducibility => the data
constrain the map; low => the model is guessing (data uninformative at its resolution). This is
fully blind (no ground truth). We check it ranks species like the known truth-Pearson does.

Run on sesame: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 venvs/fastrho/bin/python scripts/gof_repro.py
"""
import os, glob, json
import numpy as np
import sys
sys.path.insert(0, "/home/kkor/fastrho")
from scipy.stats import pearsonr, spearmanr
from fastrho.translate import load_model, predict_map_from_genotype_matrix, predict_intervals
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig
from fastrho.preprocess import mean_rate_between
from realdata_infer import get_ck, DOG_RADII

HAP = "/home/kkor/realdata/hap"
MAPS = "/home/kkor/realdata/maps"
DEV = "cuda:0"; W = 100000


def infer_map(gm, pos, mu, model_name, model, cfg, stats, edges):
    if model_name in ("gt", "dogmodel"):
        from fastcxt.sfs import basic_filtering
        fc = FeatureConfig(ld_radii=DOG_RADII) if model_name == "dogmodel" else FeatureConfig()
        gmf, posf = basic_filtering(gm.astype(np.int8), pos)
        pred = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=DEV,
                                 featurizer=GTTokenFeaturizer(config=fc, fold=True))
    else:
        pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                                mutation_rate=mu, Ne=None, device=DEV)
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    return mean_rate_between(bp, pred["r_per_bp"], edges)


def repro(key):
    z = np.load(os.path.join(HAP, key + ".npz"), allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
    m = np.load(os.path.join(MAPS, key + ".npz")); model_name = str(m["model"]); truth_p = float(m["pearson"])
    ckpt, stats_p = get_ck(model_name); model, cfg, stats = load_model(ckpt, stats_p, device=DEV)
    n = gm.shape[0]
    rng = np.arange(n)                                   # deterministic split (no RNG -> reproducible)
    h1, h2 = rng[: n // 2], rng[n // 2:]
    lo = int(pos[0]); hi = int(pos[-1]); edges = np.append(np.arange(lo, hi, W), hi)
    m1 = infer_map(gm[h1], pos, mu, model_name, model, cfg, stats, edges)
    m2 = infer_map(gm[h2], pos, mu, model_name, model, cfg, stats, edges)
    k = min(len(m1), len(m2)); a, b = m1[:k], m2[:k]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    rep = pearsonr(np.log(a[ok]), np.log(b[ok]))[0]      # half-vs-half reproducibility
    return dict(key=key, model=model_name, truth=truth_p, repro=float(rep), n_win=int(ok.sum()))


if __name__ == "__main__":
    keys = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(MAPS, "*.npz")))]
    rows = []
    for k in keys:
        try:
            rows.append(repro(k))
        except Exception as e:
            print("skip", k, repr(e)[:120])
    print("\n%-8s %-9s %8s %10s %6s" % ("species", "model", "truth_r", "repro", "nwin"))
    for r in rows:
        print("%-8s %-9s %8.3f %10.3f %6d" % (r["key"], r["model"], r["truth"], r["repro"], r["n_win"]))
    if len(rows) > 2:
        t = [r["truth"] for r in rows]; s = [r["repro"] for r in rows]
        print("\nReproducibility-vs-truth Spearman: %.3f  Pearson: %.3f  (want > 0)"
              % (spearmanr(s, t)[0], pearsonr(s, t)[0]))
    json.dump({r["key"]: r for r in rows},
              open("/home/kkor/realdata/gof_repro.json", "w"), indent=2)
