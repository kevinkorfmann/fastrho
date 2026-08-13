"""Verify realized structured-selfing simulation priors without model or map data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def summarize(path: Path, expected: int) -> dict:
    files = sorted(path.glob("ts_*.npz"))
    if len(files) != expected:
        raise RuntimeError(f"{path}: found {len(files)} shards, expected {expected}")
    metadata = []
    for shard in files:
        with np.load(shard, allow_pickle=True) as archive:
            row = json.loads(str(archive["meta"]))
        if row.get("cross_map_used") is not False:
            raise RuntimeError(f"cross-map provenance is not false in {shard}")
        if row["n_haplotypes"] != sum(row["sample_counts_by_deme"]):
            raise RuntimeError(f"haplotype-count mismatch in {shard}")
        rho_total = 4 * row["Ne"] * row["mean_rate"] * row["sequence_length"]
        if rho_total > row["rho_total_cap"] * (1 + 1e-10):
            raise RuntimeError(f"rho-total cap violated in {shard}: {rho_total}")
        row["rho_total"] = rho_total
        metadata.append(row)

    def quantiles(key: str) -> dict[str, float]:
        values = np.asarray([row[key] for row in metadata], float)
        return {
            "min": float(values.min()),
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }

    return {
        "n_shards": len(metadata),
        "design_counts": dict(sorted(Counter(row["design"] for row in metadata).items())),
        "n_haplotype_counts": {
            str(key): value
            for key, value in sorted(Counter(row["n_haplotypes"] for row in metadata).items())
        },
        "n_deme_counts": {
            str(key): value
            for key, value in sorted(Counter(row["n_demes"] for row in metadata).items())
        },
        "selfing": quantiles("selfing"),
        "effective_Ne": quantiles("Ne"),
        "effective_mean_rate": quantiles("mean_rate"),
        "rho_total": quantiles("rho_total"),
        "num_sites": quantiles("num_sites"),
        "Ne_cap_applied_fraction": float(
            np.mean([bool(row["Ne_cap_applied"]) for row in metadata])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--train", type=int, required=True)
    parser.add_argument("--val", type=int, required=True)
    parser.add_argument("--audit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "uses_cross_map": False,
        "splits": {
            "train": summarize(args.shard_root / "train", args.train),
            "val": summarize(args.shard_root / "val", args.val),
            "audit": summarize(args.shard_root / "audit", args.audit),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
