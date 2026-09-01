#!/usr/bin/env python3
"""Score ReLERNN predictions against truth averaged in each native output window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_rate_between(position: np.ndarray, rate: np.ndarray, edges: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    rate = np.asarray(rate, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if position.ndim != 1 or rate.ndim != 1 or position.size != rate.size + 1:
        raise ValueError("truth map must have len(position) == len(rate) + 1")
    if np.any(np.diff(position) <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("truth positions and scoring edges must increase strictly")
    if edges[0] < position[0] or edges[-1] > position[-1]:
        raise ValueError("native prediction window extends beyond the truth map")
    cumulative = np.r_[0.0, np.cumsum(np.diff(position) * rate)]
    integrated = np.interp(edges, position, cumulative)
    return np.diff(integrated) / np.diff(edges)


def chromosome_index(value: str) -> int:
    match = re.search(r"chr(\d+)", value)
    if match is None:
        raise ValueError(f"Cannot map ReLERNN chromosome label to a region: {value}")
    return int(match.group(1)) - 1


def parse_prediction(path: Path) -> dict[int, list[tuple[float, float, float]]]:
    records: dict[int, list[tuple[float, float, float]]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "start", "end", "recombRate"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected ReLERNN prediction header in {path}")
        for row in reader:
            index = chromosome_index(row["chrom"])
            start = float(row["start"])
            end = float(row["end"])
            prediction = float(row["recombRate"])
            if not end > start:
                raise ValueError(f"Non-positive native window in {path}: {row}")
            records.setdefault(index, []).append((start, end, prediction))
    for values in records.values():
        values.sort()
    return records


def score_arm(label: str, arm: Path, use_bias_corrected: bool = False) -> dict:
    manifest_path = arm / "relernn_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prediction_key = "bias_corrected_prediction" if use_bias_corrected else "prediction"
    if prediction_key not in manifest:
        raise KeyError(f"{prediction_key} is absent from {manifest_path}")
    prediction_path = Path(manifest[prediction_key])
    if not prediction_path.is_file():
        prediction_path = arm / "relernn_project" / "combined.PREDICT.txt"
    records = parse_prediction(prediction_path)
    predicted: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for index, rows in sorted(records.items()):
        truth_path = arm / f"region_{index:03d}.npz"
        if not truth_path.is_file():
            raise FileNotFoundError(truth_path)
        with np.load(truth_path, allow_pickle=True) as source:
            position = np.asarray(source["map_position"], dtype=float)
            rate = np.asarray(source["map_rate"], dtype=float)
        edges = np.array([rows[0][0], *[row[1] for row in rows]], dtype=float)
        starts = np.array([row[0] for row in rows], dtype=float)
        if not np.allclose(starts, edges[:-1]):
            raise ValueError(f"Native windows are not contiguous in {prediction_path}")
        truth.append(mean_rate_between(position, rate, edges))
        predicted.append(np.array([row[2] for row in rows], dtype=float))
        lengths.append(np.diff(edges))
    observed = np.concatenate(predicted)
    expected = np.concatenate(truth)
    width = np.concatenate(lengths)
    keep = np.isfinite(observed) & np.isfinite(expected) & (expected > 0) & (observed >= 0)
    observed = observed[keep]
    expected = expected[keep]
    width = width[keep]
    if observed.size < 3:
        raise ValueError(f"Too few native windows for {label}")
    pearson = float(np.corrcoef(observed, expected)[0, 1])
    spearman = float(spearmanr(observed, expected).statistic)
    positive = observed > 0
    result = {
        "label": label,
        "arm": str(arm),
        "prediction_sha256": sha256(prediction_path),
        "prediction_stage": "BSCORRECT" if use_bias_corrected else "PREDICT",
        "manifest_sha256": sha256(manifest_path),
        "training_demography": manifest["training_demography"],
        "phased": "--phased" in manifest["simulate_command"]
        and "--phased" in manifest["predict_command"],
        "assumed_mu": float(
            manifest["simulate_command"][manifest["simulate_command"].index("--assumedMu") + 1]
        ),
        "upper_rho_theta_ratio": manifest["upper_rho_theta_ratio"],
        "max_sites": manifest["max_sites"],
        "native_window_bp": {
            "minimum": int(width.min()),
            "median": float(np.median(width)),
            "maximum": int(width.max()),
        },
        "metrics": {
            "pearson": pearson,
            "spearman": spearman,
            "median_estimated_true": float(np.median(observed / expected)),
            "rmse": float(np.sqrt(np.mean((observed - expected) ** 2))),
            "n_windows": int(observed.size),
            "n_zero_predictions": int((observed == 0).sum()),
            "pearson_positive_predictions_only": (
                float(np.corrcoef(observed[positive], expected[positive])[0, 1])
                if int(positive.sum()) >= 3
                else None
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--use-bias-corrected",
        action="store_true",
        help="score ReLERNN_BSCORRECT output rather than the raw PREDICT output",
    )
    args = parser.parse_args()
    arms = [
        score_arm(label, Path(path).resolve(), args.use_bias_corrected)
        for label, path in args.arm
    ]
    output = {
        "schema_version": 1,
        "endpoint": "Pearson correlation after averaging the exact truth in each native ReLERNN output window",
        "zero_prediction_rule": "retain finite zero predictions as valid estimator outputs",
        "prediction_stage": "BSCORRECT" if args.use_bias_corrected else "PREDICT",
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
