"""Batch matched canid inference while loading the frozen checkpoint only once."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    code = os.environ.get("FASTRHO_CODE", "/home/kkor/fastrho")
    sys.path.insert(0, code)
    sys.path.insert(0, os.path.join(code, "scripts"))
    from fastcxt.sfs import basic_filtering
    from fastrho.gt_features import GTTokenFeaturizer
    from fastrho.translate import load_model, predict_intervals
    from wolf_subset_infer import (WINDOWS, feature_config, score_windows, truth_map)

    root = Path(args.root)
    model, config, stats = load_model(args.checkpoint, args.stats, device=args.device)
    featurizer = GTTokenFeaturizer(config=feature_config(stats), fold=True)
    for index in args.indices:
        for species in ("wolf", "dog"):
            key = f"{species}_match_s{index:02d}"
            output = root / "maps" / f"{key}.npz"
            if output.exists():
                print(f"skip {key}", flush=True)
                continue
            archive = np.load(root / "hap" / f"{key}.npz", allow_pickle=True)
            gm = archive["gm"]
            pos = archive["pos"].astype(float)
            filtered, filtered_pos = basic_filtering(gm.astype(np.int8), pos)
            prediction = predict_intervals(
                model, config, stats, filtered, filtered_pos, float(archive["mu"]),
                Ne=None, device=args.device, featurizer=featurizer)
            pred_pos = np.r_[prediction["pos_left"][0], prediction["pos_right"]]
            pred_rate = prediction["r_per_bp"]
            truth_pos, truth_rate = truth_map(str(archive["map_sp"]),
                                              str(archive["map_id"]),
                                              str(archive["chrom"]))
            lo, hi = int(prediction["pos_left"][0]), int(prediction["pos_right"][-1])
            scores = []
            saved = None
            for window in WINDOWS:
                result, edges, pred, truth, ok = score_windows(
                    pred_pos, pred_rate, truth_pos, truth_rate, lo, hi, window)
                scores.append(result)
                if window == 100_000:
                    saved = (edges, pred, truth, ok)
            edges, pred, truth, ok = saved
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output, centers=(edges[:-1] + edges[1:]) / 2e6,
                     pred=pred, truth=truth, ok=ok, scores_json=json.dumps(scores),
                     chrom=str(archive["chrom"]), n_hap=gm.shape[0], n_snp=gm.shape[1],
                     checkpoint=args.checkpoint, stats_path=args.stats,
                     Ne_est=float(prediction["Ne_estimated"]))
            with open(str(output) + ".json", "w") as handle:
                json.dump({"key": key, "n_hap": int(gm.shape[0]),
                           "n_snp": int(gm.shape[1]),
                           "Ne_estimated": float(prediction["Ne_estimated"]),
                           "scores": scores}, handle, indent=2)
                handle.write("\n")
            primary = scores[0]
            print(f"{key}: n={primary['n']} r={primary.get('pearson', float('nan')):.3f} "
                  f"log-r={primary.get('log_pearson', float('nan')):.3f}", flush=True)


if __name__ == "__main__":
    main()
