#!/usr/bin/env python3
"""Matched local fastrho--pyrho comparison for representative Phase 2 cohorts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import spearmanr

from fastrho.preprocess import mean_rate_between
from fastrho.translate import load_model, predict_map_from_genotype_matrix

MU = 3.5e-9
N_DIPLOID = 10
N_HAP = 20
SEED = 7_302_026
WINDOW_BP = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_vcf(gm: np.ndarray, pos: np.ndarray, path: Path, arm: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(f"##contig=<ID={arm}>\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(f"s{i}" for i in range(N_DIPLOID)) + "\n")
        last = 0
        for site, coordinate in enumerate(pos.astype(int)):
            coordinate = max(coordinate, last + 1)
            last = coordinate
            hap = gm[:, site]
            genotype = "\t".join(f"{hap[2*i]}|{hap[2*i+1]}" for i in range(N_DIPLOID))
            handle.write(f"{arm}\t{coordinate}\t.\tA\tT\t.\tPASS\t.\tGT\t{genotype}\n")


def pyrho_map(pyrho: Path, gm: np.ndarray, pos: np.ndarray, arm: str, ne: float, threads: int):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        vcf, table, rmap = root / "input.vcf", root / "lookup.hdf", root / "output.rmap"
        write_vcf(gm, pos, vcf, arm)
        commands = [
            [str(pyrho), "make_table", "-n", str(N_HAP), "-m", str(MU), "-p", str(ne),
             "--approx", "-N", str(N_HAP + 5), "--numthreads", str(threads), "-o", str(table)],
            [str(pyrho), "optimize", "--vcffile", str(vcf), "--tablefile", str(table),
             "--ploidy", "1", "-w", "50", "-bpen", "25", "--numthreads", str(threads),
             "-o", str(rmap)],
        ]
        logs = []
        for command in commands:
            run = subprocess.run(command, text=True, capture_output=True)
            logs.append({"command": command, "stdout_tail": run.stdout[-1000:], "stderr_tail": run.stderr[-1000:]})
            if run.returncode:
                raise RuntimeError(f"pyrho failed: {run.stderr[-4000:]}")
        return np.loadtxt(rmap, ndmin=2), logs


def bin_rmap(rows: np.ndarray, grid: np.ndarray) -> np.ndarray:
    left, right, rate = rows[:, -3], rows[:, -2], rows[:, -1]
    output = np.full(len(grid) - 1, np.nan)
    for index, (start, end) in enumerate(zip(grid[:-1], grid[1:])):
        overlap = np.clip(np.minimum(right, end) - np.maximum(left, start), 0, None)
        if overlap.sum() > 0:
            output[index] = np.sum(overlap * rate) / overlap.sum()
    return output


def rebin_windows(starts: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Overlap-weight source windows onto an independently anchored grid."""
    if len(starts) != len(values) or len(starts) == 0:
        raise ValueError("invalid source-window arrays")
    nominal_width = float(np.median(np.diff(starts))) if len(starts) > 1 else WINDOW_BP
    right = np.r_[starts[1:], starts[-1] + nominal_width]
    output = np.full(len(grid) - 1, np.nan)
    for index, (start, end) in enumerate(zip(grid[:-1], grid[1:])):
        overlap = np.clip(np.minimum(right, end) - np.maximum(starts, start), 0, None)
        keep = (overlap > 0) & np.isfinite(values)
        if np.any(keep):
            output[index] = np.sum(overlap[keep] * values[keep]) / np.sum(overlap[keep])
    return output


def correlation(left: np.ndarray, right: np.ndarray) -> list[float | int]:
    keep = np.isfinite(left) & np.isfinite(right) & (left > 0) & (right > 0)
    result = spearmanr(np.log(left[keep]), np.log(right[keep]))
    return [float(result.statistic), float(result.pvalue), int(np.sum(keep))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--pyrho", type=Path, required=True)
    parser.add_argument("--cohorts", default="gamb_BF,colu_CI,gamb_UG")
    parser.add_argument("--arm", default="3R")
    parser.add_argument("--start-mb", type=float, default=6)
    parser.add_argument("--end-mb", type=float, default=14)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model, config, stats = load_model(args.checkpoint, args.stats, device=args.device)
    rows, arrays = [], {}
    start_bp, end_bp = int(args.start_mb * 1e6), int(args.end_mb * 1e6)
    grid = np.arange(start_bp, end_bp + WINDOW_BP, WINDOW_BP)
    for cohort_index, cohort in enumerate(args.cohorts.split(",")):
        source = args.normalized / f"{cohort}__{args.arm}.h5"
        with h5py.File(source, "r") as handle:
            full_gm = np.asarray(handle["gm"], dtype=np.int8)
            full_pos = np.asarray(handle["pos"], dtype=np.int64)
            sample_ids = np.asarray(handle["sample_id"]).astype(str)
        rng = np.random.default_rng(SEED)
        individual = np.sort(rng.choice(len(sample_ids), N_DIPLOID, replace=False))
        haplotype = np.ravel(np.column_stack((2 * individual, 2 * individual + 1)))
        region = (full_pos >= start_bp) & (full_pos < end_bp)
        gm = full_gm[haplotype][:, region]
        pos = full_pos[region]
        segregating = (gm.sum(axis=0) > 0) & (gm.sum(axis=0) < N_HAP)
        gm, pos = gm[:, segregating], pos[segregating]
        harmonic = sum(1 / index for index in range(1, N_HAP))
        ne = max(gm.shape[1] / (harmonic * (end_bp - start_bp)) / (4 * MU), 10_000)
        prediction = predict_map_from_genotype_matrix(
            gm, pos.astype(float), model, config, stats,
            mutation_rate=MU, Ne=None, device=args.device,
        )
        breakpoints = np.r_[prediction["pos_left"][0], prediction["pos_right"]]
        matched = mean_rate_between(breakpoints, prediction["rho_per_bp"], grid)
        pyrho_rows, logs = pyrho_map(args.pyrho, gm, pos, args.arm, ne, args.threads)
        pyrho = bin_rmap(pyrho_rows, grid)
        with np.load(args.maps / f"{cohort}__{args.arm}.npz", allow_pickle=True) as published_map:
            starts = published_map["starts_100000"].astype(float)
            values = published_map["rho_100000"].astype(float)
        published = rebin_windows(starts, values, grid)
        row = {
            "cohort": cohort,
            "arm": args.arm,
            "region_mb": [args.start_mb, args.end_mb],
            "n_hap": N_HAP,
            "n_snp": int(gm.shape[1]),
            "watterson_ne": ne,
            "selected_sample_ids": sample_ids[individual].tolist(),
            "spearman_matched": correlation(matched, pyrho),
            "spearman_published": correlation(published, pyrho),
            "source_sha256": sha256(source),
            "pyrho_logs": logs,
        }
        rows.append(row)
        arrays[f"{cohort}_fastrho_matched"] = matched
        arrays[f"{cohort}_fastrho_published"] = published
        arrays[f"{cohort}_pyrho"] = pyrho
        print(cohort, row["spearman_matched"], flush=True)
    result = {
        "schema_version": 1,
        "variant": "phase2",
        "release": "Ag1000G Phase 2 AR1",
        "rows": rows,
        "mean_spearman_matched": float(np.mean([row["spearman_matched"][0] for row in rows])),
        "mean_spearman_published": float(np.mean([row["spearman_published"][0] for row in rows])),
        "checkpoint_sha256": sha256(args.checkpoint),
        "stats_sha256": sha256(args.stats),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    arrays["grid_starts"] = grid[:-1]
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
