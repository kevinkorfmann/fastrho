"""Build matched-sample empirical LD-decay data for the canid figure."""

from __future__ import annotations

import argparse
import json

import numpy as np


EDGES_BP = np.asarray([
    200, 500, 1_000, 2_000, 5_000, 10_000, 20_000,
    50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000,
])


def load_dosage(path: str, individuals: np.ndarray | None = None):
    archive = np.load(path)
    dosage = (archive["gm"][0::2] + archive["gm"][1::2]).astype(np.float32)
    if individuals is not None:
        dosage = dosage[individuals]
    frequency = dosage.mean(axis=0) / 2
    common = (frequency >= 0.05) & (frequency <= 0.95)
    return archive["pos"][common], dosage[:, common]


def mean_r2(pos, dosage, pairs_per_bin: int, rng: np.random.Generator):
    values = []
    counts = []
    for low, high in zip(EDGES_BP[:-1], EDGES_BP[1:]):
        left = rng.integers(0, len(pos), pairs_per_bin)
        target = pos[left] + rng.uniform(low, high, pairs_per_bin)
        right = np.searchsorted(pos, target)
        valid = (right < len(pos)) & (right > left)
        candidate = np.flatnonzero(valid)
        valid[candidate] &= (pos[right[candidate]] - pos[left[candidate]]) < high
        left, right = left[valid], right[valid]
        x, y = dosage[:, left], dosage[:, right]
        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)
        denominator = (x * x).sum(axis=0) * (y * y).sum(axis=0)
        variable = denominator > 0
        r2 = (x * y).sum(axis=0)[variable] ** 2 / denominator[variable]
        values.append(float(r2.mean()))
        counts.append(int(r2.size))
    return np.asarray(values), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dog", required=True)
    parser.add_argument("--wolf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--dog-subsamples", type=int, default=20)
    parser.add_argument("--pairs-per-bin", type=int, default=40_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    wolf_archive = np.load(args.wolf)
    dog_archive = np.load(args.dog)
    wolf_n = wolf_archive["gm"].shape[0] // 2
    dog_n = dog_archive["gm"].shape[0] // 2
    if dog_n < wolf_n:
        raise ValueError("dog panel is smaller than wolf panel")

    wolf_pos, wolf_dosage = load_dosage(args.wolf)
    wolf_r2, wolf_pair_counts = mean_r2(
        wolf_pos, wolf_dosage, args.pairs_per_bin * 5, rng)
    dog_replicates = []
    dog_pair_counts = []
    for _ in range(args.dog_subsamples):
        individuals = np.sort(rng.choice(dog_n, wolf_n, replace=False))
        dog_pos, dog_dosage = load_dosage(args.dog, individuals)
        estimate, counts = mean_r2(
            dog_pos, dog_dosage, args.pairs_per_bin, rng)
        dog_replicates.append(estimate)
        dog_pair_counts.append(counts)
    dog_replicates = np.asarray(dog_replicates)

    result = {
        "description": (
            "Composite-genotype LD decay over chromosome 1 positions 0-40 Mb. "
            "Each dog replicate randomly subsamples 33 of 67 individuals to match "
            "the 33-wolf panel; plotted dog values are means across replicates."
        ),
        "distance_low_bp": EDGES_BP[:-1].tolist(),
        "distance_high_bp": EDGES_BP[1:].tolist(),
        "distance_mid_bp": np.sqrt(EDGES_BP[:-1] * EDGES_BP[1:]).tolist(),
        "dog_mean_r2": dog_replicates.mean(axis=0).tolist(),
        "dog_ci95_low": np.quantile(dog_replicates, 0.025, axis=0).tolist(),
        "dog_ci95_high": np.quantile(dog_replicates, 0.975, axis=0).tolist(),
        "wolf_r2": wolf_r2.tolist(),
        "dog_n_total": int(dog_n),
        "matched_n_individuals": int(wolf_n),
        "dog_subsamples": args.dog_subsamples,
        "pairs_per_dog_bin": args.pairs_per_bin,
        "pairs_per_wolf_bin": args.pairs_per_bin * 5,
        "dog_pair_counts": dog_pair_counts,
        "wolf_pair_counts": wolf_pair_counts,
        "maf_range": [0.05, 0.95],
        "seed": args.seed,
        "source_archives": [args.dog, args.wolf],
    }
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
