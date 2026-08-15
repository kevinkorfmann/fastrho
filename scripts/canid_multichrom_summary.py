"""Aggregate empirical canid map recovery across chromosomes 1--5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCALES = (100_000, 200_000, 500_000, 1_000_000)
METRICS = ("pearson", "log_pearson", "spearman")


def scores(path: str) -> dict:
    archive = np.load(path, allow_pickle=True)
    return {int(row["window_bp"]): row for row in json.loads(str(archive["scores_json"]))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wolf", action="append", required=True)
    parser.add_argument("--dog", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    if len(args.wolf) != len(args.dog) or len(args.wolf) < 2:
        raise ValueError("provide paired wolf and dog maps for at least two chromosomes")
    wolf = [scores(path) for path in args.wolf]
    dog = [scores(path) for path in args.dog]
    rng = np.random.default_rng(args.seed)
    result = {"description": (
        "Empirical wolf and village-dog recovery against the Campbell CanFam3.1 pedigree map "
        "across chromosomes; uncertainty resamples chromosomes as paired blocks."
    ), "n_chromosomes": len(wolf), "wolf_sources": args.wolf, "dog_sources": args.dog,
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed, "scales": {}}
    for scale in SCALES:
        level = {}
        for metric in METRICS:
            w = np.asarray([entry[scale][metric] for entry in wolf], dtype=float)
            d = np.asarray([entry[scale][metric] for entry in dog], dtype=float)
            indices = rng.integers(0, len(w), (args.bootstrap, len(w)))
            draws = np.mean(w[indices] - d[indices], axis=1)
            level[metric] = {
                "wolf_mean": float(w.mean()), "dog_mean": float(d.mean()),
                "wolf_per_chromosome": w.tolist(), "dog_per_chromosome": d.tolist(),
                "wolf_minus_dog_mean": float(np.mean(w - d)),
                "wolf_minus_dog_ci95_chromosome_bootstrap": np.quantile(
                    draws, [0.025, 0.975]).tolist(),
                "bootstrap_fraction_le_zero": float(np.mean(draws <= 0)),
            }
        result["scales"][str(scale)] = level
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
