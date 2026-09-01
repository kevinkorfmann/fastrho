#!/usr/bin/env python3
"""Score all methods in exact, optionally complete, ReLERNN native windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from scipy.stats import pearsonr, spearmanr


GRID = 25_000
SCENARIOS = ("constant", "bottleneck", "expansion", "decode", "hapmap", "dog")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_rate(position: np.ndarray, rate: np.ndarray, start: float, end: float) -> float:
    position = np.asarray(position, dtype=float)
    rate = np.asarray(rate, dtype=float)
    if position.ndim != 1 or rate.ndim != 1 or rate.size + 1 != position.size:
        raise ValueError("Malformed piecewise-constant rate map")
    if not 0 <= start < end <= position[-1]:
        raise ValueError(f"Interval {start}-{end} lies outside rate map")
    cumulative = np.r_[0.0, np.cumsum(np.diff(position) * rate)]
    values = np.interp([start, end], position, cumulative)
    return float((values[1] - values[0]) / (end - start))


def chromosome_number(label: str) -> int:
    match = re.search(r"chr(\d+)", label)
    if match is None:
        raise ValueError(f"Cannot parse chromosome label {label}")
    return int(match.group(1))


def region_name(label: str) -> str:
    return f"region_{chromosome_number(label) - 1:03d}"


def parse_window_sizes(path: Path) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7:
                raise ValueError(f"Malformed windowSizes row {line_number} in {path}")
            chrom, coordinates = fields[0].split(":", maxsplit=1)
            start_text, end_text = coordinates.split("-", maxsplit=1)
            name = region_name(chrom)
            if name in result:
                raise ValueError(f"Duplicate window-size row for {name}")
            result[name] = {
                "chrom": chrom,
                "genome_start": int(start_text),
                "genome_end": int(end_text),
                "n_haplotypes": int(fields[1]),
                "nominal_window_bp": int(fields[2]),
                "minimum_sites": int(fields[3]),
                "mean_sites": int(fields[4]),
                "maximum_sites": int(fields[5]),
                "reported_window_count": int(fields[6]),
            }
    if not result:
        raise ValueError(f"No rows in {path}")
    return result


def parse_relernn(path: Path) -> Dict[str, List[dict]]:
    records: Dict[str, List[dict]] = defaultdict(list)
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "start", "end", "recombRate"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected ReLERNN header in {path}: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):
            name = region_name(row["chrom"])
            start = float(row["start"])
            end = float(row["end"])
            if not start < end:
                raise ValueError(f"Invalid ReLERNN row {row_number}: {start}-{end}")
            records[name].append(
                {
                    "start": start,
                    "end": end,
                    "n_sites": int(row["nSites"]) if row.get("nSites") else None,
                    "estimate": float(row["recombRate"]),
                }
            )
    for values in records.values():
        values.sort(key=lambda item: (item["start"], item["end"]))
    return dict(records)


def safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    if kind == "pearson":
        return float(pearsonr(x, y).statistic)
    return float(spearmanr(x, y).statistic)


def metrics(predicted: Iterable[float], expected: Iterable[float]) -> dict:
    raw = np.asarray(list(predicted), dtype=float)
    truth_raw = np.asarray(list(expected), dtype=float)
    if raw.shape != truth_raw.shape:
        raise ValueError("Prediction and truth arrays have different shapes")
    finite_prediction = np.isfinite(raw)
    finite_truth = np.isfinite(truth_raw)
    keep = finite_prediction & finite_truth & (truth_raw > 0)
    observed = raw[keep]
    truth = truth_raw[keep]
    if observed.size == 0:
        raise ValueError("No finite positive-truth observations to score")
    n_negative = int((observed < 0).sum())
    observed = np.maximum(observed, 0.0)
    positive = observed > 0
    residual = observed - truth
    result = {
        "pearson": safe_correlation(observed, truth, "pearson"),
        "spearman": safe_correlation(observed, truth, "spearman"),
        "n": int(observed.size),
        "n_input": int(raw.size),
        "n_nonfinite_predictions": int((~finite_prediction).sum()),
        "n_nonfinite_truth": int((~finite_truth).sum()),
        "n_nonpositive_truth": int((finite_truth & (truth_raw <= 0)).sum()),
        "n_negative_predictions_before_clamp": n_negative,
        "n_zero_predictions": int((observed == 0).sum()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "truth_mean": float(np.mean(truth)),
        "truth_sd": float(np.std(truth)),
        "truth_cv": float(np.std(truth) / np.mean(truth)),
    }
    if int(positive.sum()) >= 3:
        ratios = observed[positive] / truth[positive]
        result["pearson_positive_predictions_only"] = safe_correlation(
            observed[positive], truth[positive], "pearson"
        )
        result["bias_ratio_positive"] = float(np.median(ratios))
    return result


def expected_complete_intervals(sequence_length: int, nominal_width: int) -> set:
    return {
        (float(start), float(start + nominal_width))
        for start in range(0, sequence_length - nominal_width + 1, nominal_width)
    }


def score_config(
    config_dir: Path,
    prediction_dir: Path,
    method_prediction_dirs: Dict[str, Path],
    stage: str,
    complete_windows_only: bool,
) -> dict:
    config = json.loads((config_dir / "config.json").read_text())
    prediction_name = (
        "combined.PREDICT.BSCORRECTED.txt" if stage == "bscorrect" else "combined.PREDICT.txt"
    )
    prediction_path = prediction_dir / "relernn_project" / prediction_name
    windows = parse_relernn(prediction_path)
    window_sizes_path = prediction_dir / "relernn_project" / "networks" / "windowSizes.txt"
    window_sizes = parse_window_sizes(window_sizes_path)
    if set(window_sizes) != set(windows):
        raise ValueError(
            f"Window-size/PREDICT chromosome mismatch: {set(window_sizes) ^ set(windows)}"
        )

    other = {}
    other_paths = {}
    for method in ("fastrho", "pyrho"):
        path = method_prediction_dirs.get(method, prediction_dir) / f"pred_{method}.npz"
        if not path.is_file():
            path = config_dir / f"pred_{method}.npz"
        if path.is_file():
            other[method] = np.load(path, allow_pickle=True)
            other_paths[method] = path

    methods = ["relernn", *other]
    accumulated = {method: ([], []) for method in methods}
    per_region = {
        method: defaultdict(lambda: ([], []))
        for method in methods
    }
    retained_widths: List[float] = []
    all_widths: List[float] = []
    n_excluded_terminal = 0
    n_missing_complete = 0
    n_regions = 0
    sequence_length = int(config["seq_len"])
    grid_position = np.append(np.arange(0, sequence_length, GRID), sequence_length)
    truth_checksums = {}
    per_region_window_counts = {}

    for name, spec in sorted(window_sizes.items()):
        rows = windows[name]
        nominal = int(spec["nominal_window_bp"])
        if int(spec["genome_end"] - spec["genome_start"]) != sequence_length:
            raise ValueError(f"Genome length mismatch for {name}")
        expected = expected_complete_intervals(sequence_length, nominal)
        observed_intervals = {(row["start"], row["end"]) for row in rows}
        missing = expected - observed_intervals
        n_missing_complete += len(missing)
        if missing:
            raise ValueError(f"Missing complete ReLERNN windows for {name}: {sorted(missing)}")

        truth_path = config_dir / f"{name}.npz"
        if not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        for method, archive in other.items():
            if name not in archive.files:
                raise ValueError(f"Missing {method} prediction for {name}")
        n_regions += 1
        truth_checksums[name] = sha256(truth_path)
        with np.load(truth_path, allow_pickle=True) as archive:
            truth_position = np.asarray(archive["map_position"], dtype=float)
            truth_rate = np.asarray(archive["map_rate"], dtype=float)

        retained_here = 0
        excluded_here = 0
        for row in rows:
            start, end = row["start"], row["end"]
            if not 0 <= start < end <= sequence_length:
                raise ValueError(f"Invalid native interval {name}: {start}-{end}")
            width = end - start
            all_widths.append(width)
            complete = math.isclose(width, nominal, rel_tol=0.0, abs_tol=0.5)
            if complete_windows_only and not complete:
                n_excluded_terminal += 1
                excluded_here += 1
                continue
            retained_here += 1
            retained_widths.append(width)
            target = mean_rate(truth_position, truth_rate, start, end)
            accumulated["relernn"][0].append(row["estimate"])
            accumulated["relernn"][1].append(target)
            per_region["relernn"][name][0].append(row["estimate"])
            per_region["relernn"][name][1].append(target)
            for method, archive in other.items():
                rate = np.asarray(archive[name], dtype=float)
                if rate.size + 1 != grid_position.size:
                    raise ValueError(f"Unexpected {method} grid length for {name}")
                estimate = mean_rate(grid_position, rate, start, end)
                accumulated[method][0].append(estimate)
                accumulated[method][1].append(target)
                per_region[method][name][0].append(estimate)
                per_region[method][name][1].append(target)
        per_region_window_counts[name] = {
            "nominal_window_bp": nominal,
            "retained": retained_here,
            "excluded_terminal": excluded_here,
            "prediction_rows": len(rows),
        }

    retained = np.asarray(retained_widths, dtype=float)
    all_width = np.asarray(all_widths, dtype=float)
    if retained.size == 0:
        raise ValueError("No ReLERNN windows retained")
    method_results = {}
    for method, (predicted, expected) in accumulated.items():
        result = metrics(predicted, expected)
        region_prediction = []
        region_truth = []
        for name in sorted(per_region[method]):
            values, targets = per_region[method][name]
            values_array = np.asarray(values, dtype=float)
            targets_array = np.asarray(targets, dtype=float)
            keep = np.isfinite(values_array) & np.isfinite(targets_array)
            if np.any(keep):
                region_prediction.append(float(np.mean(np.maximum(values_array[keep], 0.0))))
                region_truth.append(float(np.mean(targets_array[keep])))
        region_prediction_array = np.asarray(region_prediction)
        region_truth_array = np.asarray(region_truth)
        result["region_mean_pearson"] = safe_correlation(
            region_prediction_array, region_truth_array, "pearson"
        )
        result["region_mean_spearman"] = safe_correlation(
            region_prediction_array, region_truth_array, "spearman"
        )
        result["n_region_means"] = int(region_prediction_array.size)
        method_results[method] = result

    for archive in other.values():
        archive.close()
    truth_values = np.asarray(accumulated["relernn"][1], dtype=float)
    finite_truth_values = truth_values[np.isfinite(truth_values) & (truth_values > 0)]
    assumed_mu = float(config["mu"])
    truth_quantiles = {
        f"q{int(quantile * 1000):03d}": float(np.quantile(finite_truth_values, quantile))
        for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)
    }
    return {
        "config": config["name"],
        "stage": stage,
        "complete_windows_only": complete_windows_only,
        "input_directory": str(config_dir.resolve()),
        "prediction_directory": str(prediction_dir.resolve()),
        "sequence_length_bp": sequence_length,
        "n_regions": n_regions,
        "n_prediction_windows": int(all_width.size),
        "n_retained_windows": int(retained.size),
        "n_excluded_terminal_windows": n_excluded_terminal,
        "n_missing_complete_windows": n_missing_complete,
        "native_window_bp": {
            "minimum": int(retained.min()),
            "median": float(np.median(retained)),
            "maximum": int(retained.max()),
        },
        "all_prediction_window_bp": {
            "minimum": int(all_width.min()),
            "median": float(np.median(all_width)),
            "maximum": int(all_width.max()),
        },
        "native_truth_rate": {
            "minimum": float(finite_truth_values.min()),
            "maximum": float(finite_truth_values.max()),
            "quantiles": truth_quantiles,
            "assumed_mu": assumed_mu,
            "fraction_above_default_upper_rho_theta_ratio_1": float(
                np.mean(finite_truth_values > assumed_mu)
            ),
            "q990_rho_theta_ratio": float(truth_quantiles["q990"] / assumed_mu),
            "q999_rho_theta_ratio": float(truth_quantiles["q999"] / assumed_mu),
        },
        "per_region_window_counts": per_region_window_counts,
        "checksums": {
            "relernn_prediction_sha256": sha256(prediction_path),
            "window_sizes_sha256": sha256(window_sizes_path),
            "truth_sha256": truth_checksums,
            **{
                f"{method}_prediction_sha256": sha256(path)
                for method, path in other_paths.items()
            },
        },
        "methods": method_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--prediction-root",
        type=Path,
        help="Optional isolated root containing alternate method outputs",
    )
    parser.add_argument(
        "--fastrho-root",
        type=Path,
        help="Optional isolated root containing fastrho outputs only",
    )
    parser.add_argument(
        "--pyrho-root",
        type=Path,
        help="Optional isolated root containing pyrho outputs only",
    )
    parser.add_argument("--stage", choices=("raw", "bscorrect"), required=True)
    parser.add_argument("--complete-windows-only", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prediction_root = args.prediction_root or args.root
    results = {
        "schema_version": 2,
        "stage": args.stage,
        "complete_windows_only": args.complete_windows_only,
        "scenarios": {},
    }
    for name in SCENARIOS:
        results["scenarios"][name] = score_config(
            args.root / "arms" / name,
            prediction_root / "arms" / name,
            {
                "fastrho": (args.fastrho_root or prediction_root) / "arms" / name,
                "pyrho": (args.pyrho_root or prediction_root) / "arms" / name,
            },
            args.stage,
            args.complete_windows_only,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
