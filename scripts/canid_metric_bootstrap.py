"""Block-bootstrap chromosome-wide canid map correlations across fixed metrics."""

from __future__ import annotations

import argparse
import json

import numpy as np


def aggregate(values: np.ndarray, factor: int) -> np.ndarray:
    size = len(values) // factor * factor
    return np.asarray(values[:size], float).reshape(-1, factor).mean(axis=1)


def summarize(first: np.ndarray, second: np.ndarray, metric: str) -> np.ndarray:
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > 0)
    first, second = first[valid], second[valid]
    if metric == "log_pearson":
        first, second = np.log(first), np.log(second)
    return np.array([
        len(first), first.sum(), second.sum(), np.square(first).sum(),
        np.square(second).sum(), np.multiply(first, second).sum(),
    ])


def correlation(summary: np.ndarray) -> np.ndarray:
    n, sx, sy, sxx, syy, sxy = np.moveaxis(np.asarray(summary, float), -1, 0)
    numerator = n * sxy - sx * sy
    denominator = np.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    return numerator / denominator


def block_summaries(archive, factor: int, metric: str, n: int) -> np.ndarray:
    summaries = []
    for start in range(0, n, 50):
        block = np.arange(start, min(start + 50, n))
        first = aggregate(archive["pred"][block], factor)
        second = aggregate(archive["truth"][block], factor)
        summaries.append(summarize(first, second, metric))
    return np.asarray(summaries)


def compare(wolf, dog, factor: int, metric: str, rng, replicates: int):
    n = min(len(wolf["pred"]), len(dog["pred"]))
    wolf_summaries = block_summaries(wolf, factor, metric, n)
    dog_summaries = block_summaries(dog, factor, metric, n)
    observed = float(correlation(wolf_summaries.sum(axis=0))
                     - correlation(dog_summaries.sum(axis=0)))
    selected = rng.integers(0, len(wolf_summaries), (replicates, len(wolf_summaries)))
    wolf_draws = correlation(wolf_summaries[selected].sum(axis=1))
    dog_draws = correlation(dog_summaries[selected].sum(axis=1))
    draws = wolf_draws - dog_draws
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
        "description": "Wolf-minus-dog differences from paired 5-Mb block bootstrap",
        "seed": args.seed,
        "replicates": args.replicates,
        "block_size_bp": 5_000_000,
        "source_wolf_map": args.wolf,
        "comparisons": {},
    }
    for label, path in args.dog:
        dog = np.load(path)
        result["comparisons"][label] = {"source_dog_map": path}
        for metric in ("pearson", "log_pearson"):
            result["comparisons"][label][metric] = {
                "100kb": compare(wolf, dog, 1, metric, rng, args.replicates),
                "1mb": compare(wolf, dog, 10, metric, rng, args.replicates),
            }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
