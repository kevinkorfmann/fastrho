"""Evaluation for fastrho.

Metrics mirror pyrho's framing (Pearson / Spearman / log-Pearson at multiple bp scales,
L2 / log-L2) plus an absolute-calibration bias ratio, 95% CI coverage, and hotspot AUPRC.
Runs directly on held-out feature shards (which carry the exact per-interval true rate).
"""

from __future__ import annotations

import argparse
import glob
import json

import numpy as np


def score_rates(pred: np.ndarray, true: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    m = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    p, t = pred[m], true[m]
    if len(p) < 3:
        return {"n": int(m.sum())}
    return {
        "pearson": float(pearsonr(p, t)[0]),
        "spearman": float(spearmanr(p, t)[0]),
        "log_pearson": float(pearsonr(np.log(p), np.log(t))[0]),
        "l2": float(np.sqrt(np.mean((p - t) ** 2))),
        "log_l2": float(np.sqrt(np.mean((np.log(p) - np.log(t)) ** 2))),
        "bias_ratio": float(np.median(p / t)),     # ~1.0 = good absolute calibration
        "n": int(m.sum()),
    }


def _rebin(positions, interval_rate, window_size):
    from fastrho.preprocess import mean_rate_between
    L = positions[-1]
    edges = np.append(np.arange(positions[0], L, window_size), L)
    centers = edges[:-1]
    return centers, mean_rate_between(positions, interval_rate, edges)


def absolute_rate_view(pred: dict, Ne: float) -> dict:
    """Return absolute rates and intervals conditional on a specified positive ``Ne``."""
    if not np.isfinite(Ne) or Ne <= 0:
        raise ValueError("Ne must be finite and positive")
    scale = 4.0 * float(Ne)
    return {
        "r_per_bp": np.asarray(pred["rho_per_bp"]) / scale,
        "r_ci_lo": np.asarray(pred["rho_ci_lo"]) / scale,
        "r_ci_hi": np.asarray(pred["rho_ci_hi"]) / scale,
        "Ne_used": float(Ne),
    }


def evaluate_dir(checkpoint, stats_path, shard_dir, device="cuda:0",
                 scales=(10000, 100000), max_shards=None, hotspot_scale=10000,
                 hotspot_fold=2.0, ne_mode="true"):
    if ne_mode not in {"true", "estimated", "both"}:
        raise ValueError("ne_mode must be 'true', 'estimated', or 'both'")
    from fastrho.translate import load_model, predict_from_tokens
    model, cfg, stats = load_model(checkpoint, stats_path, device=device)

    files = sorted(glob.glob(f"{shard_dir}/ts_*.npz"))
    if max_shards:
        files = files[:max_shards]

    modes = ("true", "estimated") if ne_mode == "both" else (ne_mode,)
    pools = {}
    coverage = {}
    hp_true = {}
    hp_score = {}
    ne_pairs = []
    for mode in modes:
        pools[mode] = {"interval": {"p": [], "t": []}}
        for window in scales:
            pools[mode][f"{window // 1000}kb"] = {"p": [], "t": []}
        coverage[mode] = []
        hp_true[mode] = []
        hp_score[mode] = []

    used = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "tokens" not in z or z["tokens"].shape[0] < 3:
            continue
        meta = json.loads(str(z["meta"]))
        Ne = meta.get("Ne")
        positions = z["positions"]
        true_r = z["interval_target"].astype(float)            # (S-1,)
        pred = predict_from_tokens(
            model,
            cfg,
            stats,
            z["tokens"],
            positions,
            int(meta.get("n_haplotypes", 2 * int(meta["n_samples"]))),
            float(meta["mutation_rate"]),
            Ne=None,
            device=device,
        )
        if Ne:
            ne_pairs.append((float(Ne), float(pred["Ne_estimated"])))
        views = {"estimated": pred}
        if Ne:
            views["true"] = absolute_rate_view(pred, float(Ne))
        for mode in modes:
            if mode not in views:
                continue
            view = views[mode]
            pr = view["r_per_bp"]
            pools[mode]["interval"]["p"].append(pr)
            pools[mode]["interval"]["t"].append(true_r)
            coverage[mode].append(
                np.mean((true_r >= view["r_ci_lo"]) & (true_r <= view["r_ci_hi"]))
            )
            for window in scales:
                _, pb = _rebin(positions, pr, window)
                _, tb = _rebin(positions, true_r, window)
                pools[mode][f"{window // 1000}kb"]["p"].append(pb)
                pools[mode][f"{window // 1000}kb"]["t"].append(tb)
            _, tb_h = _rebin(positions, true_r, hotspot_scale)
            _, pb_h = _rebin(positions, pr, hotspot_scale)
            hp_true[mode].append((tb_h > hotspot_fold * np.median(tb_h)).astype(int))
            hp_score[mode].append(pb_h)
        used += 1

    results = {}
    for mode in modes:
        mode_result = {
            scale: score_rates(np.concatenate(data["p"]), np.concatenate(data["t"]))
            for scale, data in pools[mode].items()
            if data["p"]
        }
        if coverage[mode]:
            mode_result["coverage95"] = float(np.mean(coverage[mode]))
        try:
            from sklearn.metrics import average_precision_score
            if hp_true[mode]:
                labels = np.concatenate(hp_true[mode])
                scores = np.concatenate(hp_score[mode])
                if labels.sum() > 0:
                    mode_result["hotspot_auprc"] = float(
                        average_precision_score(labels, scores)
                    )
        except ImportError:
            pass
        mode_result["n_shards"] = len(pools[mode]["interval"]["p"])
        results[mode] = mode_result

    if ne_pairs:
        true_ne, estimated_ne = np.asarray(ne_pairs, dtype=float).T
        results["Ne_estimation"] = {
            "n_shards": len(ne_pairs),
            "median_estimated_over_true": float(np.median(estimated_ne / true_ne)),
            "log_rmse": float(np.sqrt(np.mean((np.log(estimated_ne) - np.log(true_ne)) ** 2))),
        }
    results["n_shards_seen"] = used
    return results[modes[0]] if ne_mode != "both" else results


def _print(results: dict):
    if "true" in results or "estimated" in results:
        for mode in ("true", "estimated"):
            if mode in results:
                print(f"\nAbsolute-rate evaluation conditional on {mode} Ne")
                _print(results[mode])
        if "Ne_estimation" in results:
            ne = results["Ne_estimation"]
            print(
                "\nNe auxiliary head: "
                f"median estimated/true={ne['median_estimated_over_true']:.3f}, "
                f"log-RMSE={ne['log_rmse']:.3f} (n={ne['n_shards']})"
            )
        return
    print(f"\n=== fastrho evaluation ({results.get('n_shards')} shards) ===")
    order = ["interval", "10kb", "100kb"]
    keys = [k for k in order if k in results] + \
           [k for k in results if k not in order and isinstance(results[k], dict)]
    for sc in keys:
        m = results[sc]
        if not isinstance(m, dict):
            continue
        print(f"[{sc:>9}] pearson={m.get('pearson', float('nan')):.3f} "
              f"spearman={m.get('spearman', float('nan')):.3f} "
              f"log_pearson={m.get('log_pearson', float('nan')):.3f} "
              f"log_l2={m.get('log_l2', float('nan')):.3f} "
              f"bias={m.get('bias_ratio', float('nan')):.2f} n={m.get('n')}")
    if "coverage95" in results:
        print(f"95% CI coverage: {results['coverage95']:.3f}  (target 0.95)")
    if "hotspot_auprc" in results:
        print(f"hotspot AUPRC (10kb): {results['hotspot_auprc']:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate a fastrho checkpoint on held-out shards")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stats", required=True, help="feat_stats.npz")
    ap.add_argument("--shards", required=True, help="dir of held-out ts_*.npz feature shards")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--ne-mode", choices=("true", "estimated", "both"), default="both",
                    help="score absolute rates with simulated true Ne, estimated Ne, or both")
    args = ap.parse_args()
    res = evaluate_dir(args.checkpoint, args.stats, args.shards,
                       device=args.device, max_shards=args.max_shards, ne_mode=args.ne_mode)
    _print(res)


if __name__ == "__main__":
    main()
