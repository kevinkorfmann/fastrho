"""Blind GoF via fastrho's OWN calibrated uncertainty.

fastrho emits a calibrated per-interval log-rho sigma (beta-NLL head; held-out cov95~0.95).
A calibrated model trained on a prior reports INFLATED uncertainty when the data looks unlike
its training distribution (out-of-distribution: selfer, extreme-Ne, unphased short-LD) -- exactly
the regimes where the map is untrustworthy. So mean predicted uncertainty is a no-ground-truth
reliability signal. We check it ranks species the same way the known truth-Pearson does.

Run on sesame: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 venvs/fastrho/bin/python scripts/gof_unc.py
"""
import os, glob
import numpy as np
import sys
sys.path.insert(0, "/home/kkor/fastrho")
from scipy.stats import spearmanr
from fastrho.translate import load_model, predict_map_from_genotype_matrix, predict_intervals
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig
from fastrho.preprocess import mean_rate_between
from realdata_infer import get_ck, DOG_RADII

HAP = "/home/kkor/realdata/hap"
MAPS = "/home/kkor/realdata/maps"
DEV = "cuda:0"
W = 100000


def uncert(key):
    z = np.load(os.path.join(HAP, key + ".npz"), allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); mu = float(z["mu"])
    m = np.load(os.path.join(MAPS, key + ".npz"))
    model_name = str(m["model"]); truth_p = float(m["pearson"])
    ckpt, stats_p = get_ck(model_name)
    model, cfg, stats = load_model(ckpt, stats_p, device=DEV)
    if model_name in ("gt", "dogmodel"):
        from fastcxt.sfs import basic_filtering
        fc = FeatureConfig(ld_radii=DOG_RADII) if model_name == "dogmodel" else FeatureConfig()
        gmf, posf = basic_filtering(gm.astype(np.int8), pos)
        pred = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=DEV,
                                 featurizer=GTTokenFeaturizer(config=fc, fold=True))
    else:
        pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                                mutation_rate=mu, Ne=None, device=DEV)
    s = pred["sigma_log_rho"]                       # calibrated per-interval log-rho sigma
    sig = s[np.isfinite(s)]
    # WITHIN-species reliability test: bin sigma to the same 100kb windows as the saved map,
    # then correlate window sigma with the window's actual |log error| vs truth.
    left = pred["pos_left"]; right = pred["pos_right"]
    bp = np.r_[left[0], right]
    lo = int(left[0]); hi = int(right[-1])
    edges = np.append(np.arange(lo, hi, W), hi)
    sig_w = mean_rate_between(bp, s, edges)          # mean sigma per window
    pred_w = m["pred"].astype(float); truth_w = m["truth"].astype(float)
    k = min(len(sig_w), len(pred_w))
    sw, pw, tw = sig_w[:k], pred_w[:k], truth_w[:k]
    ok = np.isfinite(sw) & np.isfinite(pw) & np.isfinite(tw) & (pw > 0) & (tw > 0)
    err = np.abs(np.log(pw[ok]) - np.log(tw[ok]))    # per-window |log error|
    within = spearmanr(sw[ok], err)[0] if ok.sum() > 10 else float("nan")
    return dict(key=key, model=model_name, truth=truth_p,
                mean_sigma=float(np.mean(sig)), med_sigma=float(np.median(sig)),
                within_sigma_vs_error=float(within), n_win=int(ok.sum()))


if __name__ == "__main__":
    keys = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(MAPS, "*.npz")))]
    rows = []
    for k in keys:
        try:
            rows.append(uncert(k))
        except Exception as e:
            print("skip", k, repr(e)[:120])
    print("\n%-8s %-9s %8s %9s %16s" % ("species", "model", "truth_r", "mean_sig", "within_sig_vs_err"))
    for r in rows:
        print("%-8s %-9s %8.3f %9.3f %16.3f"
              % (r["key"], r["model"], r["truth"], r["mean_sigma"], r["within_sigma_vs_error"]))
    if len(rows) > 2:
        t = [r["truth"] for r in rows]; s = [r["mean_sigma"] for r in rows]
        # CROSS-species: does mean sigma rank accuracy? (expect >0; we find it does NOT)
        print("\nCROSS-species reliability skill = -Spearman(mean_sigma, truth_r): %.3f  (want>0)"
              % (-spearmanr(s, t)[0]))
        w = [r["within_sigma_vs_error"] for r in rows]
        # WITHIN-species: does sigma flag which windows are wrong? (the genuinely useful claim)
        print("WITHIN-species sigma-vs-error (mean over species): %.3f  (want>0: sigma flags errors)"
              % np.nanmean(w))
    import json
    json.dump({r["key"]: r for r in rows},
              open("/home/kkor/realdata/gof_uncertainty.json", "w"), indent=2)
