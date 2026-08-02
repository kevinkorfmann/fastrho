"""Strict head-to-head scoring on shared held-out shards.

External methods have environment- and dataset-specific pipelines, which live in
``scripts/``. This module does not pretend to run those pipelines. Instead, it
scores their committed per-region prediction archives against exactly the same
truth arrays used for fastrho and rejects missing or misaligned regions.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

from fastrho.evaluate import score_rates


def benchmark_fastrho(checkpoint, stats, shard_dir, device="cuda:0", max_shards=None):
    from fastrho.evaluate import evaluate_dir

    t0 = time.time()
    result = evaluate_dir(
        checkpoint,
        stats,
        shard_dir,
        device=device,
        max_shards=max_shards,
        ne_mode="both",
    )
    result["wall_clock_s"] = time.time() - t0
    result["per_dataset_training"] = False
    result["uncertainty"] = "single-pass log-rate interval conditional on Ne"
    return result


def score_prediction_archive(path, shard_dir, max_shards=None):
    """Score a ``.npz`` of per-interval predictions keyed by shard stem.

    Each key must match a ``ts_*.npz`` shard filename without its extension and
    contain one prediction per value in that shard's ``interval_target`` array.
    This deliberately strict interchange format prevents hidden interpolation,
    dropped-region, and unequal-test-set effects in method comparisons.
    """
    predictions = np.load(path, allow_pickle=False)
    files = sorted(glob.glob(os.path.join(shard_dir, "ts_*.npz")))
    if max_shards is not None:
        files = files[:max_shards]
    if not files:
        raise FileNotFoundError(f"no ts_*.npz shards under {shard_dir}")

    pred_parts = []
    true_parts = []
    used = []
    for shard_path in files:
        key = os.path.splitext(os.path.basename(shard_path))[0]
        if key not in predictions.files:
            raise KeyError(f"{path} has no prediction for {key}")
        with np.load(shard_path, allow_pickle=False) as shard:
            truth = np.asarray(shard["interval_target"], dtype=float)
        pred = np.asarray(predictions[key], dtype=float)
        if pred.shape != truth.shape:
            raise ValueError(
                f"prediction shape mismatch for {key}: {pred.shape} != {truth.shape}"
            )
        pred_parts.append(pred)
        true_parts.append(truth)
        used.append(key)

    result = score_rates(np.concatenate(pred_parts), np.concatenate(true_parts))
    result.update(
        status="scored",
        n_shards=len(used),
        shard_ids=used,
        prediction_archive=os.path.basename(path),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Score fastrho and external methods fairly")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--pyrho-predictions")
    parser.add_argument("--relernn-predictions")
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "fastrho": benchmark_fastrho(
            args.checkpoint,
            args.stats,
            args.shards,
            device=args.device,
            max_shards=args.max_shards,
        ),
    }
    for name, path in (
        ("pyrho", args.pyrho_predictions),
        ("relernn", args.relernn_predictions),
    ):
        report[name] = (
            score_prediction_archive(path, args.shards, args.max_shards)
            if path
            else {"status": "not supplied"}
        )
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
