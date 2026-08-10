"""Block-bootstrap empirical wolf-transfer differences against dog LD maps."""

from __future__ import annotations

import argparse
import json

import numpy as np


def correlation(first, second):
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    valid = (np.isfinite(first) & np.isfinite(second)
             & (first > 0) & (second > 0))
    return float(np.corrcoef(first[valid], second[valid])[0, 1])


def aggregate(values, factor):
    size = len(values) // factor * factor
    return np.asarray(values[:size]).reshape(-1, factor).mean(axis=1)


def compare(wolf, dog, factor, rng, replicates):
    n = min(len(wolf["pred"]), len(dog["pred"]))
    blocks = [np.arange(start, min(start + 50, n))
              for start in range(0, n, 50)]
    observed = (
        correlation(aggregate(wolf["pred"][:n], factor),
                    aggregate(wolf["truth"][:n], factor))
        - correlation(aggregate(dog["pred"][:n], factor),
                      aggregate(dog["truth"][:n], factor))
    )
    draws = np.empty(replicates)
    for replicate in range(replicates):
        selected = rng.integers(0, len(blocks), len(blocks))
        wolf_pred, wolf_truth, dog_pred, dog_truth = [], [], [], []
        for index in selected:
            block = blocks[index]
            wolf_pred.extend(aggregate(wolf["pred"][block], factor))
            wolf_truth.extend(aggregate(wolf["truth"][block], factor))
            dog_pred.extend(aggregate(dog["pred"][block], factor))
            dog_truth.extend(aggregate(dog["truth"][block], factor))
        draws[replicate] = (correlation(wolf_pred, wolf_truth)
                            - correlation(dog_pred, dog_truth))
    return {
        "difference": observed,
        "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "bootstrap_fraction_le_zero": float(np.mean(draws <= 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wolf", required=True)
    parser.add_argument("--dog", action="append", nargs=2, required=True,
                        metavar=("LABEL", "MAP"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    wolf = np.load(args.wolf)
    rng = np.random.default_rng(args.seed)
    result = {
        "description": "Wolf-minus-dog Pearson differences from 5-Mb block bootstrap",
        "seed": args.seed,
        "replicates": args.replicates,
        "block_size_bp": 5_000_000,
        "source_wolf_map": args.wolf,
        "comparisons": {},
    }
    for label, path in args.dog:
        dog = np.load(path)
        result["comparisons"][label] = {
            "source_dog_map": path,
            "100kb": compare(wolf, dog, 1, rng, args.replicates),
            "1mb": compare(wolf, dog, 10, rng, args.replicates),
        }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
