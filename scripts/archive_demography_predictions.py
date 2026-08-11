#!/usr/bin/env python3
"""Archive every paired demographic prediction on the common 25-kb grid."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCENARIOS = ("bottleneck", "expansion")
METHODS = ("relernn", "pyrho")
HISTORIES = ("constant", "matched")
GRID_BP = 25_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_rate_between(position: np.ndarray, rate: np.ndarray, edges: np.ndarray) -> np.ndarray:
    position = np.asarray(position, dtype=float)
    rate = np.asarray(rate, dtype=float)
    cumulative = np.r_[0.0, np.cumsum(np.diff(position) * rate)]
    integrated = np.interp(edges, position, cumulative)
    return np.diff(integrated) / np.diff(edges)


def archive(root: Path, output: Path) -> dict:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict = {"schema_version": 1, "grid_bp": GRID_BP, "arms": {}}
    for scenario in SCENARIOS:
        reference = root / "arms" / f"{scenario}_pyrho_constant"
        config = json.loads((reference / "config.json").read_text())
        edges = np.arange(0, int(config["seq_len"]) + GRID_BP, GRID_BP, dtype=int)
        regions = sorted(path.stem for path in reference.glob("region_*.npz"))
        for region in regions:
            with np.load(reference / f"{region}.npz", allow_pickle=True) as source:
                truth = mean_rate_between(source["map_position"], source["map_rate"], edges)
            arrays[f"truth__{scenario}__{region}"] = truth

        for method in METHODS:
            for history in HISTORIES:
                name = f"{scenario}_{method}_{history}"
                path = root / "arms" / name / f"pred_{method}.npz"
                with np.load(path, allow_pickle=True) as predictions:
                    prediction_regions = sorted(
                        key for key in predictions.files if not key.startswith("_")
                    )
                    if prediction_regions != regions:
                        raise ValueError(f"Prediction regions differ for {name}")
                    for region in regions:
                        values = np.asarray(predictions[region], dtype=float)
                        truth = arrays[f"truth__{scenario}__{region}"]
                        if values.shape != truth.shape or not np.all(np.isfinite(values)):
                            raise ValueError(f"Invalid 25-kb prediction for {name}/{region}")
                        arrays[f"pred__{name}__{region}"] = values
                metadata["arms"][name] = {
                    "method": method,
                    "history": history,
                    "scenario": scenario,
                    "n_regions": len(regions),
                    "n_windows": len(regions) * (len(edges) - 1),
                }

        manifest = json.loads((root / "fastrho_reference.json").read_text())
        name = f"{scenario}_fastrho_fixed"
        path = root / manifest["scenarios"][scenario]["file"]
        if sha256(path) != manifest["scenarios"][scenario]["sha256"]:
            raise ValueError(f"Fixed fastrho prediction hash differs for {scenario}")
        with np.load(path) as predictions:
            prediction_regions = sorted(key for key in predictions.files if not key.startswith("_"))
            if prediction_regions != regions:
                raise ValueError(f"Prediction regions differ for {name}")
            for region in regions:
                values = np.asarray(predictions[region], dtype=float)
                truth = arrays[f"truth__{scenario}__{region}"]
                if values.shape != truth.shape or not np.all(np.isfinite(values)):
                    raise ValueError(f"Invalid 25-kb prediction for {name}/{region}")
                arrays[f"pred__{name}__{region}"] = values
        metadata["arms"][name] = {
            "method": "fastrho",
            "history": "fixed",
            "scenario": scenario,
            "n_regions": len(regions),
            "n_windows": len(regions) * (len(edges) - 1),
        }

    arrays["_metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metadata = archive(args.root.resolve(), args.output.resolve())
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
