#!/usr/bin/env python3
"""Score the canonical fixed-model predictions on all frozen benchmark regions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fastrho.evaluate import score_rates
from fastrho.preprocess import mean_rate_between

SCENARIOS = ("bottleneck", "expansion")
GRID_BP = 25_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    length = (len(values) // factor) * factor
    return values[:length].reshape(-1, factor).mean(axis=1)


def score(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    output: dict = {
        "schema_version": 1,
        "reference_manifest_sha256": sha256(manifest_path),
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        record = manifest["scenarios"][scenario]
        prediction_path = root / record["file"]
        if sha256(prediction_path) != record["sha256"]:
            raise ValueError(f"Fixed fastrho prediction hash differs for {scenario}")
        reference = root / "arms" / f"{scenario}_pyrho_constant"
        config = json.loads((reference / "config.json").read_text())
        edges = np.arange(0, int(config["seq_len"]) + GRID_BP, GRID_BP)
        regions = sorted(path.stem for path in reference.glob("region_*.npz"))
        with np.load(prediction_path) as predictions:
            prediction_regions = sorted(key for key in predictions.files if not key.startswith("_"))
            if prediction_regions != regions:
                raise ValueError(f"Fixed fastrho prediction regions differ for {scenario}")
            truth, inferred = [], []
            for region in regions:
                with np.load(reference / f"{region}.npz", allow_pickle=True) as source:
                    true_rate = mean_rate_between(source["map_position"], source["map_rate"], edges)
                prediction = np.asarray(predictions[region], dtype=float)
                if prediction.shape != true_rate.shape or not np.all(np.isfinite(prediction)):
                    raise ValueError(f"Invalid fixed fastrho prediction for {scenario}/{region}")
                truth.append(true_rate)
                inferred.append(prediction)
        scenario_result = {"n_regions": len(regions), "prediction_sha256": record["sha256"]}
        for scale, factor in (("25kb", 1), ("100kb", 4)):
            pooled_truth = np.concatenate([block_mean(values, factor) for values in truth])
            pooled_inferred = np.concatenate([block_mean(values, factor) for values in inferred])
            scenario_result[scale] = score_rates(pooled_inferred, pooled_truth)
        output["scenarios"][scenario] = scenario_result
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = score(args.root.resolve(), args.manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
