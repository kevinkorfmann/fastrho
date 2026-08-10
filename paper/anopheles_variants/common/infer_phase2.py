#!/usr/bin/env python3
"""Run the frozen high-Ne fastrho checkpoint on normalized Phase 2 panels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from fastrho.preprocess import mean_rate_between
from fastrho.translate import load_model, predict_map_from_genotype_matrix

MU = 3.5e-9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, config, feature_stats = load_model(args.checkpoint, args.stats, device=args.device)
    sources = sorted(args.input.glob("*.h5"))[args.shard :: args.nshards]
    for source in sources:
        target = args.out / f"{source.stem}.npz"
        if target.exists():
            print(f"skip {target}")
            continue
        with h5py.File(source, "r") as handle:
            matrix = np.asarray(handle["gm"], dtype=np.int8)
            positions = np.asarray(handle["pos"], dtype=np.float64)
            cohort = str(handle.attrs["cohort"])
            arm = str(handle.attrs["arm"])
        started = time.time()
        prediction = predict_map_from_genotype_matrix(
            matrix,
            positions,
            model,
            config,
            feature_stats,
            mutation_rate=MU,
            Ne=None,
            device=args.device,
        )
        left = prediction["pos_left"]
        right = prediction["pos_right"]
        breakpoints = np.r_[left[0], right]
        output: dict[str, object] = {
            "pop": cohort,
            "region": arm,
            "n_hap": matrix.shape[0],
            "n_snp": matrix.shape[1],
            "Ne_est": float(prediction["Ne_estimated"]),
            "seconds": time.time() - started,
            "release": "Ag1000G Phase 2 AR1",
        }
        for window in (100_000, 50_000):
            edges = np.arange(left[0], right[-1], window)
            edges = np.append(edges, right[-1])
            output[f"starts_{window}"] = edges[:-1]
            output[f"rho_{window}"] = mean_rate_between(breakpoints, prediction["rho_per_bp"], edges)
            output[f"r_{window}"] = mean_rate_between(breakpoints, prediction["r_per_bp"], edges)
        np.savez(target, **output)
        summary = {
            "cohort": cohort,
            "arm": arm,
            "n_hap": matrix.shape[0],
            "n_snp": matrix.shape[1],
            "Ne_est": output["Ne_est"],
            "seconds": output["seconds"],
        }
        target.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{cohort} {arm}: {matrix.shape[1]:,} SNPs -> {target}")


if __name__ == "__main__":
    main()
