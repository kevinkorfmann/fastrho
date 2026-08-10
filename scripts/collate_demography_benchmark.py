#!/usr/bin/env python3
"""Collate the frozen constant-versus-matched competitor benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

SCENARIOS = ("bottleneck", "expansion")
METHODS = ("relernn", "pyrho")
HISTORIES = ("constant", "matched")
SCALES = ("25kb", "100kb")
METRICS = ("pearson", "spearman", "bias_ratio", "n")
GRID_BP = 25_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relernn_native_windows(root: Path, name: str) -> dict[str, float | int]:
    path = root / "arms" / name / "relernn_project" / "networks" / "windowSizes.txt"
    values = [int(line.split()[2]) for line in path.read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError(f"No ReLERNN native windows recorded in {path}")
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "n_regions": len(values),
    }


def _mean_rate_between(position: np.ndarray, rate: np.ndarray, edges: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    rate = np.asarray(rate, dtype=float)
    cumulative = np.r_[0.0, np.cumsum(np.diff(position) * rate)]
    integrated = np.interp(edges, position, cumulative)
    return np.diff(integrated) / np.diff(edges)


def _block_mean(values: np.ndarray, block_size: int) -> np.ndarray:
    if values.size % block_size:
        raise ValueError(f"Window count {values.size} is not divisible by {block_size}")
    return values.reshape(-1, block_size).mean(axis=1)


def paired_metrics(root: Path, scenario: str, method: str) -> dict[str, dict[str, dict]]:
    """Score both histories on their fixed, jointly valid window support."""

    from fastrho.evaluate import score_rates

    reference = root / "arms" / f"{scenario}_pyrho_constant"
    config = json.loads((reference / "config.json").read_text())
    edges = np.append(np.arange(0, int(config["seq_len"]), GRID_BP), int(config["seq_len"]))
    truth_paths = sorted(reference.glob("region_*.npz"))
    if not truth_paths:
        raise FileNotFoundError(f"No truth maps found in {reference}")
    predictions = {
        history: np.load(
            root / "arms" / f"{scenario}_{method}_{history}" / f"pred_{method}.npz",
            allow_pickle=False,
        )
        for history in HISTORIES
    }
    try:
        pools: dict[str, dict[str, list[np.ndarray]]] = {
            scale: {"truth": [], **{history: [] for history in HISTORIES}}
            for scale in SCALES
        }
        for truth_path in truth_paths:
            region = truth_path.stem
            with np.load(truth_path, allow_pickle=True) as source:
                truth_25kb = _mean_rate_between(source["map_position"], source["map_rate"], edges)
            raw = {history: np.asarray(predictions[history][region], float) for history in HISTORIES}
            if any(values.shape != truth_25kb.shape for values in raw.values()):
                raise ValueError(f"Prediction shape differs for {scenario}/{method}/{region}")
            for scale, block_size in (("25kb", 1), ("100kb", 4)):
                pools[scale]["truth"].append(_block_mean(truth_25kb, block_size))
                for history in HISTORIES:
                    pools[scale][history].append(_block_mean(raw[history], block_size))

        output = {history: {} for history in HISTORIES}
        for scale in SCALES:
            truth = np.concatenate(pools[scale]["truth"])
            arm_values = {
                history: np.concatenate(pools[scale][history]) for history in HISTORIES
            }
            joint = np.isfinite(truth) & (truth > 0)
            for history in HISTORIES:
                joint &= np.isfinite(arm_values[history]) & (arm_values[history] > 0)
            if int(joint.sum()) < 3:
                raise ValueError(f"Too few jointly valid windows for {scenario}/{method}/{scale}")
            for history in HISTORIES:
                score = score_rates(arm_values[history][joint], truth[joint])
                output[history][scale] = {metric: score[metric] for metric in METRICS}
        return output
    finally:
        for archive in predictions.values():
            archive.close()


def collate(root: Path, design: Path, fastrho_reference: Path | None = None) -> dict:
    output: dict = {
        "schema_version": 1,
        "design_sha256": sha256(design),
        "input_manifest_sha256": sha256(root / "input_manifest.json"),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        output["scenarios"][scenario] = {}
        for method in METHODS:
            output["scenarios"][scenario][method] = {
                "arms": {},
                "matched_minus_constant": {},
                "support_rule": (
                    "finite, strictly positive truth and predictions in both demographic arms"
                ),
            }
            rescored = paired_metrics(root, scenario, method)
            for history in HISTORIES:
                name = f"{scenario}_{method}_{history}"
                # The stage-level JSON proves that the arm finished. Final paired metrics are
                # rederived from its prediction archive on a common window mask below.
                json.loads((root / "results" / f"{name}.json").read_text())
                arm = rescored[history]
                if method == "relernn":
                    arm["native_window_bp"] = relernn_native_windows(root, name)
                output["scenarios"][scenario][method]["arms"][history] = arm
            for scale in SCALES:
                constant = output["scenarios"][scenario][method]["arms"]["constant"][scale]
                matched = output["scenarios"][scenario][method]["arms"]["matched"][scale]
                if constant["n"] != matched["n"]:
                    raise AssertionError("joint-support rescoring returned unequal window counts")
                output["scenarios"][scenario][method]["matched_minus_constant"][scale] = {
                    metric: matched[metric] - constant[metric]
                    for metric in ("pearson", "spearman", "bias_ratio")
                }
    if fastrho_reference is not None:
        reference = json.loads(fastrho_reference.read_text())
        output["fastrho_reference_sha256"] = sha256(fastrho_reference)
        for scenario in SCENARIOS:
            output["scenarios"][scenario]["fastrho_reference"] = reference["scenarios"][scenario]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fastrho-reference", type=Path)
    args = parser.parse_args()
    reference = args.fastrho_reference.resolve() if args.fastrho_reference else None
    result = collate(args.root.resolve(), args.design.resolve(), reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
