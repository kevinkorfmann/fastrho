"""Infer and score a canid panel with an explicitly supplied checkpoint.

Usage:
  python scripts/wolf_subset_infer.py KEY CHECKPOINT STATS OUT_NPZ [DEVICE]

The genotype input is ``/home/kkor/realdata/hap/KEY.npz``. Correlations are
reported against its stdpopsim reference at several window sizes so a coarse
pedigree map is not judged only at 100-kb resolution.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
import stdpopsim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastrho.gt_features import GTTokenFeaturizer
from fastrho.preprocess import mean_rate_between
from fastrho.translate import load_model, predict_intervals


WINDOWS = (100_000, 200_000, 500_000, 1_000_000, 2_000_000, 4_000_000)


def feature_config(stats):
    from fastrho.features import FeatureConfig

    kw = {"ld_radii": tuple(int(x) for x in np.asarray(
        stats.get("ld_radii", (5_000, 25_000, 50_000))).ravel())}
    for name in ("disjoint_bands", "stride_after", "max_neighbors"):
        if name in stats:
            value = int(np.asarray(stats[name]).item())
            kw[name] = bool(value) if name == "disjoint_bands" else value
    return FeatureConfig(**kw)


def truth_map(map_species, map_id, chrom):
    species = stdpopsim.get_species(map_species)
    rate_map = species.get_genetic_map(map_id).get_chromosome_map(chrom.replace("chr", ""))
    pos = np.asarray(rate_map.position, dtype=float)
    rate = np.asarray(rate_map.rate, dtype=float)
    return pos, np.where(np.isfinite(rate), rate, 0.0)


def score_windows(pred_pos, pred_rate, truth_pos, truth_rate, lo, hi, window):
    edges = np.append(np.arange(lo, hi, window), hi)
    pred = mean_rate_between(pred_pos, pred_rate, edges)
    truth = mean_rate_between(truth_pos, truth_rate, edges)
    n = min(len(pred), len(truth))
    pred, truth = pred[:n], truth[:n]
    ok = np.isfinite(pred) & np.isfinite(truth) & (pred > 0) & (truth > 0)
    result = {"window_bp": window, "n": int(ok.sum())}
    if ok.sum() >= 3:
        result.update(
            pearson=float(pearsonr(pred[ok], truth[ok]).statistic),
            spearman=float(spearmanr(pred[ok], truth[ok]).statistic),
            log_pearson=float(pearsonr(np.log(pred[ok]), np.log(truth[ok])).statistic),
        )
    return result, edges, pred, truth, ok


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    key, checkpoint, stats_path, output = sys.argv[1:5]
    device = sys.argv[5] if len(sys.argv) > 5 else "cuda:0"

    z = np.load(f"/home/kkor/realdata/hap/{key}.npz", allow_pickle=True)
    gm = z["gm"]
    pos = z["pos"].astype(np.float64)
    mu = float(z["mu"])
    chrom = str(z["chrom"])
    model, cfg, stats = load_model(checkpoint, stats_path, device=device)

    try:
        from fastcxt.sfs import basic_filtering
        gmf, posf = basic_filtering(gm.astype(np.int8), pos)
    except ImportError:
        from fastrho.filtering import basic_filtering
        gmf, posf = basic_filtering(gm.astype(np.int8), pos)

    pred = predict_intervals(
        model, cfg, stats, gmf, posf, mu, Ne=None, device=device,
        featurizer=GTTokenFeaturizer(config=feature_config(stats), fold=True),
    )
    pred_pos = np.r_[pred["pos_left"][0], pred["pos_right"]]
    pred_rate = pred["r_per_bp"]
    truth_pos, truth_rate = truth_map(str(z["map_sp"]), str(z["map_id"]), chrom)
    lo, hi = int(pred["pos_left"][0]), int(pred["pos_right"][-1])

    scores = []
    saved = None
    for window in WINDOWS:
        result, edges, pr, tr, ok = score_windows(
            pred_pos, pred_rate, truth_pos, truth_rate, lo, hi, window)
        scores.append(result)
        if window == 100_000:
            saved = (edges, pr, tr, ok)
        print(f"{key} {window // 1000:4d} kb: "
              f"r={result.get('pearson', float('nan')):.3f} "
              f"log-r={result.get('log_pearson', float('nan')):.3f} "
              f"rho={result.get('spearman', float('nan')):.3f} n={result['n']}")

    edges, pr, tr, ok = saved
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez(
        output, centers=(edges[:-1] + edges[1:]) / 2e6, pred=pr, truth=tr, ok=ok,
        scores_json=json.dumps(scores), chrom=chrom, n_hap=gm.shape[0], n_snp=gm.shape[1],
        checkpoint=checkpoint, stats_path=stats_path, Ne_est=float(pred["Ne_estimated"]),
    )
    with open(output + ".json", "w") as handle:
        json.dump({"key": key, "n_hap": int(gm.shape[0]), "n_snp": int(gm.shape[1]),
                   "Ne_estimated": float(pred["Ne_estimated"]), "scores": scores},
                  handle, indent=2)


if __name__ == "__main__":
    main()
