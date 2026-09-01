#!/usr/bin/env python3
"""Validate and collect the single-node 10-Mb staged timing benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_files(root: Path) -> list[Path]:
    files = [root / "config.json", root / "genome.bed"]
    for suffix in ("vcf", "trees", "npz"):
        files.extend(sorted(root.glob(f"region_*.{suffix}")))
    return files


def positive(value: object, label: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive, got {number}")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-arm", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    arm = root / "arm"
    source = args.source_arm.resolve()
    config = json.loads((arm / "config.json").read_text())
    expected = {
        "name": "bottleneck",
        "demography": "bottleneck",
        "seq_len": 10_000_000,
        "n_regions": 20,
        "n_dip": 10,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Unexpected config {key}: {config.get(key)!r} != {value!r}")

    for suffix in ("vcf", "trees", "npz"):
        paths = sorted(arm.glob(f"region_*.{suffix}"))
        if len(paths) != 20:
            raise ValueError(f"Expected 20 {suffix} inputs, found {len(paths)}")

    source_files = input_files(source)
    copied_files = input_files(arm)
    if [path.name for path in source_files] != [path.name for path in copied_files]:
        raise ValueError("Source and timing input file lists differ")
    checksums = {}
    for source_path, copied_path in zip(source_files, copied_files):
        source_sum = sha256(source_path)
        copied_sum = sha256(copied_path)
        if source_sum != copied_sum:
            raise ValueError(f"Input checksum mismatch for {copied_path.name}")
        checksums[copied_path.name] = copied_sum

    fastrho_path = arm / "pred_fastrho.npz"
    with np.load(fastrho_path, allow_pickle=False) as archive:
        region_keys = sorted(key for key in archive.files if key.startswith("region_"))
        if len(region_keys) != 20:
            raise ValueError(f"Expected 20 fastrho predictions, found {len(region_keys)}")
        fastrho_total = positive(archive["_wall"], "fastrho total time")
        fastrho_prediction = positive(
            archive["_prediction_wall"], "fastrho prediction time"
        )
        fastrho_load = positive(archive["_model_load_wall"], "fastrho load time")

    pyrho = json.loads((arm / "pyrho_runtime.json").read_text())
    if int(pyrho.get("successful_regions", -1)) != 20:
        raise ValueError("pyrho did not complete all 20 regions")
    if len(list(arm.glob("region_*.rmap"))) != 20:
        raise ValueError("Expected 20 pyrho map files")
    pyrho_lookup = positive(pyrho["lookup_table_wall_seconds"], "pyrho lookup time")
    pyrho_inference = positive(pyrho["optimize_wall_seconds"], "pyrho inference time")
    pyrho_total = positive(pyrho["total_wall_seconds"], "pyrho total time")

    relernn = json.loads((arm / "relernn_run_manifest.json").read_text())
    required_manifest = {
        "simulate_complete": True,
        "train_predict_complete": True,
        "max_sites": 1750,
        "upper_rho_theta_ratio": 3.0,
        "n_train": 100_000,
        "n_validation": 10_000,
        "n_test": 10_000,
        "epochs": 100,
        "simulation_seed": 1,
        "training_seed": 1,
        "training_demography": "matched",
    }
    for key, value in required_manifest.items():
        if relernn.get(key) != value:
            raise ValueError(
                f"Unexpected ReLERNN manifest {key}: {relernn.get(key)!r} != {value!r}"
            )
    prediction_path = Path(relernn["prediction"])
    if not prediction_path.is_file() or prediction_path.stat().st_size == 0:
        raise ValueError("ReLERNN prediction output is missing or empty")
    if sha256(prediction_path) != relernn["prediction_sha256"]:
        raise ValueError("ReLERNN prediction checksum mismatch")
    relernn_simulation = positive(relernn["simulate_wall_seconds"], "ReLERNN simulation time")
    relernn_training = positive(relernn["train_wall_seconds"], "ReLERNN training time")
    relernn_prediction = positive(relernn["predict_wall_seconds"], "ReLERNN prediction time")
    relernn_total = relernn_simulation + relernn_training + relernn_prediction

    gpu_text = (root / "hardware" / "gpu.csv").read_text().strip()
    hostname = (root / "hardware" / "hostname.txt").read_text().strip()
    if "B200" not in gpu_text:
        raise ValueError(f"Expected a B200 GPU, got {gpu_text!r}")

    result = {
        "schema_version": 2,
        "valid": True,
        "benchmark_input": {
            "description": "20 phased 10-Mb bottleneck regions used in the intended-regime accuracy benchmark",
            "haplotypes_per_region": 20,
            "region_length_bp": 10_000_000,
            "n_regions": 20,
            "source_arm": str(source),
            "input_checksums_sha256": checksums,
        },
        "execution": {
            "design": "sequential execution in one Slurm allocation on one physical node",
            "hostname": hostname,
            "gpu": gpu_text,
            "allocated_cpus": int((root / "hardware" / "allocated_cpus.txt").read_text()),
            "slurm_job_id": (root / "hardware" / "slurm_job_id.txt").read_text().strip(),
            "order": ["fastrho", "pyrho", "ReLERNN"],
        },
        "measurements": {
            "fastrho": {
                "prediction_or_inference_seconds": fastrho_prediction,
                "dataset_specific_workflow_seconds": fastrho_total,
                "stage_seconds": {
                    "checkpoint_load": fastrho_load,
                    "prediction": fastrho_prediction,
                },
                "training_treatment": "upstream simulation and pretraining are amortized across datasets",
            },
            "pyrho": {
                "prediction_or_inference_seconds": pyrho_inference,
                "dataset_specific_workflow_seconds": pyrho_total,
                "stage_seconds": {
                    "lookup_table": pyrho_lookup,
                    "inference": pyrho_inference,
                },
            },
            "relernn": {
                "prediction_or_inference_seconds": relernn_prediction,
                "dataset_specific_workflow_seconds": relernn_total,
                "stage_seconds": {
                    "simulation": relernn_simulation,
                    "training": relernn_training,
                    "prediction": relernn_prediction,
                },
            },
        },
        "reporting": {
            "excluded_from_all_endpoints": [
                "software installation",
                "result ingestion",
                "manuscript scoring",
            ],
            "relernn_optional_stage_excluded": "ReLERNN_BSCORRECT uncertainty and bias-correction stage",
            "use": "absolute wall-clock measurements on one identical node allocation",
        },
        "provenance": {
            "fastrho_prediction_sha256": sha256(fastrho_path),
            "pyrho_runtime_sha256": sha256(arm / "pyrho_runtime.json"),
            "relernn_manifest_sha256": sha256(arm / "relernn_run_manifest.json"),
            "relernn_prediction_sha256": sha256(prediction_path),
            "lscpu_sha256": sha256(root / "hardware" / "lscpu.txt"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["measurements"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
