#!/usr/bin/env python3
"""Apply a frozen ReLERNN model to a new, compatible VCF suite.

This helper recomputes ReLERNN's input-dependent native window sizes and VCF
shards, while reusing the simulation sets, normalization metadata, and trained
weights from a fully validated source project.  It refuses model-shape changes
and records checksums so this diagnostic cannot silently retrain or overwrite
the source project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ReLERNN.manager import Manager


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_vcfs(config_dir: Path) -> Path:
    regions = sorted(config_dir.glob("region_*.vcf"))
    if not regions:
        raise FileNotFoundError(f"No region VCFs in {config_dir}")
    combined = config_dir / "combined.vcf"
    contigs: list[str] = []
    body: list[str] = []
    sample_line: str | None = None
    for vcf in regions:
        for line in vcf.read_text().splitlines(keepends=True):
            if line.startswith("##contig"):
                contigs.append(line)
            elif line.startswith("#CHROM"):
                if sample_line is not None and line != sample_line:
                    raise ValueError("VCF sample columns differ between regions")
                sample_line = line
            elif not line.startswith("#"):
                body.append(line)
    if sample_line is None:
        raise ValueError("No #CHROM header found")
    combined.write_text(
        "##fileformat=VCFv4.2\n"
        + "".join(contigs)
        + '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        + sample_line
        + "".join(body)
    )
    return combined


def source_capacity(source_project: Path) -> int:
    maximum = 0
    for dataset in ("train", "vali", "test"):
        with (source_project / dataset / "info.p").open("rb") as handle:
            info = pickle.load(handle)
        maximum = max(maximum, max(int(value) for value in info["segSites"]))
    return maximum


def executable(name: str) -> Path:
    explicit = os.environ.get(f"{name}_EXECUTABLE")
    path = Path(explicit) if explicit else Path(sys.executable).with_name(name)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare_project(config_dir: Path, source_project: Path, n_cpu: int, seed: int) -> dict:
    project = config_dir / "relernn_project"
    if project.exists() and any(project.iterdir()):
        raise FileExistsError(f"Refusing to reuse nonempty output project: {project}")
    project.mkdir(parents=True, exist_ok=True)
    for dataset in ("train", "vali", "test"):
        source = source_project / dataset
        if not source.is_dir():
            raise FileNotFoundError(source)
        (project / dataset).symlink_to(source.resolve(), target_is_directory=True)
    network_dir = project / "networks"
    network_dir.mkdir()
    source_network = source_project / "networks"
    for path in source_network.iterdir():
        if path.name == "windowSizes.txt":
            continue
        target = network_dir / path.name
        target.symlink_to(path.resolve(), target_is_directory=path.is_dir())
    (project / "splitVCFs").mkdir()

    combined = combine_vcfs(config_dir)
    chromosomes = []
    with (config_dir / "genome.bed").open() as handle:
        for line in handle:
            chrom, start, end = line.split()
            chromosomes.append(f"{chrom}:{start}-{end}")
    manager = Manager(
        vcf=str(combined),
        mask=None,
        winSizeMx=1750,
        forceWinSize=0,
        forceDiploid=False,
        chromosomes=chromosomes,
        vcfDir=str(project / "splitVCFs"),
        projectDir=str(project),
        networkDir=str(network_dir),
        seed=seed,
    )
    manager.splitVCF(nProc=n_cpu)
    windows, n_samples, max_input_sites, _ = manager.countSites(nProc=n_cpu)
    capacity = source_capacity(source_project)
    if max_input_sites > capacity:
        raise ValueError(
            f"New input requires padding to {max_input_sites} sites, exceeding the "
            f"frozen model capacity {capacity}"
        )
    return {
        "project": project,
        "combined": combined,
        "windows": [
            [str(row[0]), *[int(value) for value in row[1:]]]
            for row in windows
        ],
        "n_samples": int(n_samples),
        "max_input_sites": int(max_input_sites),
        "source_model_capacity": int(capacity),
    }


def load_prepared_project(config_dir: Path, source_project: Path) -> dict:
    """Validate and reopen a project whose input preprocessing already completed."""

    project = config_dir / "relernn_project"
    combined = config_dir / "combined.vcf"
    window_path = project / "networks" / "windowSizes.txt"
    if not combined.is_file() or not window_path.is_file():
        raise FileNotFoundError("Prepared combined VCF or windowSizes.txt is missing")
    windows = []
    with window_path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 7:
                raise ValueError(f"Malformed prepared window row: {line.rstrip()}")
            windows.append(
                [fields[0], *[int(value) for value in fields[1:]]]
            )
    if len(windows) != 20:
        raise ValueError(f"Expected 20 prepared contigs, found {len(windows)}")
    sample_sizes = {row[1] for row in windows}
    if len(sample_sizes) != 1:
        raise ValueError(f"Prepared contigs have inconsistent sample sizes: {sample_sizes}")
    capacity = source_capacity(source_project)
    max_input_sites = max(row[5] for row in windows)
    if max_input_sites > capacity:
        raise ValueError(
            f"Prepared input requires {max_input_sites} sites, exceeding frozen "
            f"model capacity {capacity}"
        )
    for dataset in ("train", "vali", "test"):
        if not (project / dataset / "info.p").is_file():
            raise FileNotFoundError(project / dataset / "info.p")
    if not (project / "networks" / "model.weights.h5").is_file():
        raise FileNotFoundError(project / "networks" / "model.weights.h5")
    return {
        "project": project,
        "combined": combined,
        "windows": windows,
        "n_samples": sample_sizes.pop(),
        "max_input_sites": max_input_sites,
        "source_model_capacity": capacity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--source-project", required=True, type=Path)
    parser.add_argument("--n-cpu", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--resume-prepared",
        action="store_true",
        help="reuse validated VCF shards/window sizes after a prediction-only failure",
    )
    args = parser.parse_args()
    config_dir = args.config_dir.resolve()
    source_project = args.source_project.resolve()
    generation = json.loads((config_dir / "native_regime_generation.json").read_text())
    if not json.loads((config_dir / "config.json").read_text()).get("native_regime"):
        raise ValueError("Input config is not marked native_regime")
    prepared = (
        load_prepared_project(config_dir, source_project)
        if args.resume_prepared
        else prepare_project(config_dir, source_project, args.n_cpu, args.seed)
    )
    project = prepared.pop("project")
    combined = prepared.pop("combined")
    prediction_command = [
        str(executable("ReLERNN_PREDICT")),
        "--vcf",
        str(combined),
        "--projectDir",
        str(project),
        "--phased",
        "--seed",
        str(args.seed),
        "--gpuID",
        str(args.gpu_id),
    ]
    started = time.perf_counter()
    with (config_dir / "relernn_reuse_predict.log").open("w") as handle:
        subprocess.run(
            prediction_command,
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONHASHSEED": str(args.seed)},
        )
    elapsed = float(time.perf_counter() - started)
    prediction = project / "combined.PREDICT.txt"
    if not prediction.is_file():
        raise FileNotFoundError(prediction)
    manifest = {
        "schema_version": 1,
        "operation": "frozen_model_reuse_prediction",
        "config_dir": str(config_dir),
        "source_project": str(source_project),
        "native_regime_generation_sha256": sha256(
            config_dir / "native_regime_generation.json"
        ),
        "source_model_weights_sha256": sha256(
            source_project / "networks" / "model.weights.h5"
        ),
        "source_test_results_sha256": sha256(
            source_project / "networks" / "testResults.p"
        ),
        "source_window_sizes_sha256": sha256(
            source_project / "networks" / "windowSizes.txt"
        ),
        "combined_vcf_sha256": sha256(combined),
        "prediction_sha256": sha256(prediction),
        "window_sizes_sha256": sha256(project / "networks" / "windowSizes.txt"),
        "prediction_wall_seconds": elapsed,
        "prediction_command": prediction_command,
        "seed": args.seed,
        "gpu_id": args.gpu_id,
        "generation_scenario": generation["scenario"],
        **prepared,
    }
    (config_dir / "relernn_reuse_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    # Compatibility manifest allows the existing isolated bias-correction wrapper
    # to run without claiming that a second training occurred.
    (config_dir / "relernn_run_manifest.json").write_text(
        json.dumps(
            {
                "arm": generation["scenario"],
                "train_predict_complete": True,
                "training_reused": True,
                "source_project": str(source_project),
                "prediction": str(prediction),
                "prediction_sha256": sha256(prediction),
                "predict_wall_seconds": elapsed,
                "max_sites": 1750,
                "training_seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(prediction)


if __name__ == "__main__":
    main()
