"""Run frozen-model full-panel and split-panel inference across chromosomes."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


CODE = os.environ.get("FASTRHO_CODE", "/home/kkor/fastrho_dr")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))

from transect_infer import infer_unphased  # noqa: E402
from fastrho.preprocess import mean_rate_between  # noqa: E402
import realdata_infer as realdata  # noqa: E402


SCALES = (100_000, 200_000, 500_000, 1_000_000)


def split_rows(n_rows: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "haploid":
        return np.arange(0, n_rows, 2), np.arange(1, n_rows, 2)
    n_individuals = n_rows // 2
    a = np.arange(0, n_individuals, 2)
    b = np.arange(1, n_individuals, 2)
    return (np.column_stack((2 * a, 2 * a + 1)).ravel(),
            np.column_stack((2 * b, 2 * b + 1)).ravel())


def metric(first: np.ndarray, second: np.ndarray, kind: str) -> float:
    valid = np.isfinite(first) & np.isfinite(second) & (first > 0) & (second > 0)
    x, y = first[valid], second[valid]
    if len(x) < 4:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x, y).statistic)
    if kind == "log_pearson":
        return float(pearsonr(np.log(x), np.log(y)).statistic)
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(kind)


def window_map(edges: np.ndarray, positions: np.ndarray, rates: np.ndarray) -> np.ndarray:
    return mean_rate_between(positions, rates, edges)


def infer(gm, pos, mu):
    edges, rate, n_snp = infer_unphased(gm, pos, mu)
    return np.asarray(edges), np.asarray(rate), int(n_snp)


def chromosome(path: str) -> dict:
    archive = np.load(path, allow_pickle=True)
    gm = archive["gm"]
    pos = archive["pos"].astype(float)
    mu = float(archive["mu"])
    chrom = str(archive["chrom"])
    mode = str(archive["mode"])
    rows_a, rows_b = split_rows(gm.shape[0], mode)
    full_bp, full_rate, full_n = infer(gm, pos, mu)
    a_bp, a_rate, a_n = infer(gm[rows_a], pos, mu)
    b_bp, b_rate, b_n = infer(gm[rows_b], pos, mu)
    lo = int(max(full_bp[0], a_bp[0], b_bp[0]))
    hi = int(min(full_bp[-1], a_bp[-1], b_bp[-1]))
    result = {"chrom": chrom, "source": path, "mode": mode,
              "n_hap": int(gm.shape[0]), "n_snp_input": int(gm.shape[1]),
              "n_snp_used_full": full_n, "n_snp_used_half_a": a_n,
              "n_snp_used_half_b": b_n, "scales": {}}
    for scale in SCALES:
        edges = np.append(np.arange(lo, hi, scale), hi)
        full = window_map(edges, full_bp, full_rate)
        first = window_map(edges, a_bp, a_rate)
        second = window_map(edges, b_bp, b_rate)
        values = {"n_windows": int(len(full)), "split_reproducibility": {}}
        for kind in ("pearson", "log_pearson", "spearman"):
            values["split_reproducibility"][kind] = metric(first, second, kind)
        if "map_sp" in archive.files:
            truth = realdata.truth_windows(str(archive["map_sp"]), str(archive["map_id"]),
                                           chrom, edges)
            n = min(len(full), len(truth))
            values["external_map"] = {kind: metric(full[:n], truth[:n], kind)
                                      for kind in ("pearson", "log_pearson", "spearman")}
            values["map_id"] = str(archive["map_id"])
        result["scales"][str(scale)] = values
    print(f"{chrom}: {gm.shape[0]} haplotypes, {gm.shape[1]:,} SNPs", flush=True)
    return result


def aggregate(chromosomes: list[dict], replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    output = {"n_chromosomes": len(chromosomes), "bootstrap_replicates": replicates,
              "bootstrap_seed": seed, "unit": "chromosome", "scales": {}}
    for scale in SCALES:
        level = {}
        for comparison in ("split_reproducibility", "external_map"):
            available = [item for item in chromosomes
                         if comparison in item["scales"][str(scale)]]
            if not available:
                continue
            level[comparison] = {}
            for kind in ("pearson", "log_pearson", "spearman"):
                values = np.asarray([item["scales"][str(scale)][comparison][kind]
                                     for item in available], dtype=float)
                values = values[np.isfinite(values)]
                draws = np.mean(values[rng.integers(0, len(values),
                                                    (replicates, len(values)))], axis=1)
                level[comparison][kind] = {
                    "mean": float(np.mean(values)), "median": float(np.median(values)),
                    "ci95_chromosome_bootstrap": np.quantile(draws, [0.025, 0.975]).tolist(),
                    "per_chromosome": values.tolist(),
                }
        output["scales"][str(scale)] = level
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="glob for one species' chromosome NPZs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--species", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.input))
    if len(paths) < 2:
        raise ValueError(f"multi-chromosome analysis requires >=2 inputs; found {len(paths)}")
    chromosomes = [chromosome(path) for path in paths]
    payload = {"species": args.species, "description": (
        "Frozen-model full-panel recovery and disjoint-sample reproducibility across chromosomes; "
        "uncertainty resamples chromosomes."
    ), "chromosomes": chromosomes,
        "aggregate": aggregate(chromosomes, args.bootstrap, args.seed)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
