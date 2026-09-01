#!/usr/bin/env python3
"""Run one frozen ReLERNN demographic arm in separable Slurm stages."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


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
    sample_line = None
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


def auto_uprtr(config_dir: Path, mu: float) -> int:
    rates: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    for path in sorted(config_dir.glob("region_*.npz")):
        with np.load(path, allow_pickle=True) as archive:
            position = np.asarray(archive["map_position"], float)
            rate = np.asarray(archive["map_rate"], float)
        segment = np.diff(position)
        keep = np.isfinite(rate) & (segment > 0)
        rates.append(rate[keep])
        lengths.append(segment[keep])
    if not rates:
        raise FileNotFoundError("No truth-map NPZ files available for the frozen prior rule")
    rate = np.concatenate(rates)
    length = np.concatenate(lengths)
    order = np.argsort(rate)
    rate = rate[order]
    length = length[order]
    cdf = np.cumsum(length) / length.sum()
    quantile = float(rate[np.searchsorted(cdf, 0.999)])
    return int(math.ceil(1.15 * quantile / mu))


def resolve_uprtr(
    config_dir: Path, mu: float, explicit: float | None
) -> tuple[float | int, str]:
    """Resolve the rate-prior bound while recording whether it was prespecified."""

    if explicit is None:
        return auto_uprtr(config_dir, mu), "analysis_specific_length_weighted_p99.9"
    if not math.isfinite(explicit) or explicit <= 0:
        raise ValueError("--upper-rho-theta-ratio must be finite and positive")
    return explicit, "explicit_cli"


def executable(name: str) -> str:
    explicit = os.environ.get(f"{name}_EXECUTABLE")
    executable_dir = os.environ.get("RELERNN_EXECUTABLE_DIR")
    path = (
        Path(explicit)
        if explicit
        else Path(executable_dir) / name
        if executable_dir
        else Path(sys.executable).with_name(name)
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def compatibility_provenance() -> dict[str, str]:
    values = {}
    for variable in ("DEMO_RELERNN_PATCH", "DEMO_RELERNN_BSCORRECT_PATCH"):
        raw = os.environ.get(variable)
        if raw:
            path = Path(raw)
            values[f"{variable.lower()}_sha256"] = sha256(path)
    executable_dir = os.environ.get("RELERNN_EXECUTABLE_DIR")
    if executable_dir:
        values["relernn_executable_dir"] = str(Path(executable_dir).resolve())
    bscorrect_executable = os.environ.get("ReLERNN_BSCORRECT_EXECUTABLE")
    if bscorrect_executable:
        values["relernn_bscorrect_executable"] = str(
            Path(bscorrect_executable).resolve()
        )
    return values


def run(command: list[str], log: Path, env: dict[str, str]) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w") as handle:
        subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return float(time.perf_counter() - started)


def update_manifest(path: Path, values: dict) -> None:
    current = json.loads(path.read_text()) if path.is_file() else {}
    current.update(values)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")


def simulate(args: argparse.Namespace, config: dict, project: Path, env: dict[str, str]) -> None:
    if project.exists() and any(project.iterdir()):
        raise FileExistsError(f"Refusing to overwrite ReLERNN project: {project}")
    project.mkdir(parents=True, exist_ok=True)
    combined = combine_vcfs(args.config_dir)
    uprtr, uprtr_source = resolve_uprtr(
        args.config_dir,
        float(config["mu"]),
        args.upper_rho_theta_ratio,
    )
    command = [
        executable("ReLERNN_SIMULATE"),
        "--vcf",
        str(combined),
        "--genome",
        str(args.config_dir / "genome.bed"),
        "--projectDir",
        str(project),
        "--assumedMu",
        str(config["mu"]),
        "--assumedGenTime",
        "1",
        "--upperRhoThetaRatio",
        str(uprtr),
        "--nCPU",
        str(args.n_cpu),
        "--seed",
        str(args.seed),
        "--phased",
        "--maxSites",
        str(args.max_sites),
        "--nTrain",
        str(args.n_train),
        "--nVali",
        str(args.n_validation),
        "--nTest",
        str(args.n_test),
    ]
    demography = "constant"
    if args.demography_history is not None:
        command.extend(["--demographicHistory", str(args.demography_history)])
        demography = "matched"
    manifest = args.config_dir / "relernn_run_manifest.json"
    update_manifest(
        manifest,
        {
            "arm": config["name"],
            "training_demography": demography,
            "demography_history": str(args.demography_history) if args.demography_history else None,
            "demography_sha256": sha256(args.demography_history)
            if args.demography_history
            else None,
            "combined_vcf_sha256": sha256(combined),
            "upper_rho_theta_ratio": uprtr,
            "upper_rho_theta_ratio_source": uprtr_source,
            "n_train": args.n_train,
            "n_validation": args.n_validation,
            "n_test": args.n_test,
            "max_sites": args.max_sites,
            "simulation_seed": args.seed,
            "compatibility_patch_sha256": sha256(Path(os.environ["DEMO_RELERNN_PATCH"])),
            "simulate_command": command,
            **compatibility_provenance(),
        },
    )
    elapsed = run(command, args.config_dir / "relernn_sim.log", env)
    update_manifest(
        manifest,
        {"simulate_complete": True, "simulate_wall_seconds": elapsed},
    )


def train_predict(
    args: argparse.Namespace, config: dict, project: Path, env: dict[str, str]
) -> None:
    manifest = args.config_dir / "relernn_run_manifest.json"
    if not manifest.is_file() or not json.loads(manifest.read_text()).get("simulate_complete"):
        raise RuntimeError("Simulation stage is incomplete")
    combined = args.config_dir / "combined.vcf"
    train_command = [
        executable("ReLERNN_TRAIN"),
        "--projectDir",
        str(project),
        "--nEpochs",
        str(args.epochs),
        "--nCPU",
        str(args.train_cpu),
        "--seed",
        str(args.seed),
        "--gpuID",
        str(args.gpu_id),
    ]
    train_elapsed = run(train_command, args.config_dir / "relernn_train.log", env)
    predict_command = [
        executable("ReLERNN_PREDICT"),
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
    predict_elapsed = run(predict_command, args.config_dir / "relernn_predict.log", env)
    prediction = project / "combined.PREDICT.txt"
    if not prediction.is_file():
        predictions = sorted(
            path
            for path in project.glob("*.PREDICT.txt")
            if ".BSCORRECT" not in path.name
        )
        prediction = predictions[0] if predictions else prediction
    if not prediction.is_file():
        raise FileNotFoundError("ReLERNN prediction output was not created")
    update_manifest(
        manifest,
        {
            "epochs": args.epochs,
            "training_seed": args.seed,
            "train_command": train_command,
            "predict_command": predict_command,
            "train_wall_seconds": train_elapsed,
            "predict_wall_seconds": predict_elapsed,
            "prediction": str(prediction),
            "prediction_sha256": sha256(prediction),
            "train_predict_complete": True,
            **compatibility_provenance(),
        },
    )
    print(prediction)


def bias_correct(
    args: argparse.Namespace, config: dict, project: Path, env: dict[str, str]
) -> None:
    """Run ReLERNN's recommended optional bias-correction module additively."""

    manifest = args.config_dir / "relernn_run_manifest.json"
    if not manifest.is_file() or not json.loads(manifest.read_text()).get(
        "train_predict_complete"
    ):
        raise RuntimeError("Training and prediction stages are incomplete")
    # Invoke the patched script with the same interpreter that is running this
    # wrapper.  Relying on its /usr/bin/env shebang can select the container's
    # system Python, which lacks the isolated ReLERNN dependencies (notably
    # msprime) on some compute nodes.
    command = [
        sys.executable,
        executable("ReLERNN_BSCORRECT"),
        "--projectDir",
        str(project),
        "--nCPU",
        str(args.n_cpu),
        "--nSlice",
        str(args.n_slices),
        "--nReps",
        str(args.n_reps),
        "--seed",
        str(args.seed),
        "--gpuID",
        str(args.gpu_id),
    ]
    elapsed = run(command, args.config_dir / "relernn_bscorrect.log", env)
    corrected = project / "combined.PREDICT.BSCORRECTED.txt"
    if not corrected.is_file():
        candidates = sorted(project.glob("*.PREDICT.BSCORRECTED.txt"))
        corrected = candidates[0] if candidates else corrected
    if not corrected.is_file():
        raise FileNotFoundError("ReLERNN bias-corrected prediction was not created")
    update_manifest(
        manifest,
        {
            "bias_correct_command": command,
            "bias_corrected_prediction": str(corrected),
            "bias_corrected_prediction_sha256": sha256(corrected),
            "bias_correct_n_slices": args.n_slices,
            "bias_correct_n_reps": args.n_reps,
            "bias_correct_wall_seconds": elapsed,
            "bias_correct_complete": True,
            **compatibility_provenance(),
        },
    )
    print(corrected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("simulate", "train-predict", "bias-correct"))
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--demography-history", type=Path)
    parser.add_argument("--n-train", type=int, default=100000)
    parser.add_argument("--n-validation", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--max-sites", type=int, default=256)
    parser.add_argument(
        "--upper-rho-theta-ratio",
        type=float,
        help=(
            "explicit ReLERNN upperRhoThetaRatio; when omitted, retain the historical "
            "analysis-specific length-weighted p99.9 rule"
        ),
    )
    parser.add_argument("--n-cpu", type=int, default=64)
    parser.add_argument("--train-cpu", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n-slices", type=int, default=100)
    parser.add_argument("--n-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.config_dir = args.config_dir.resolve()
    if args.demography_history is not None:
        args.demography_history = args.demography_history.resolve()
        if not args.demography_history.is_file():
            raise FileNotFoundError(args.demography_history)
    config = json.loads((args.config_dir / "config.json").read_text())
    project = args.config_dir / "relernn_project"
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", str(args.seed))
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if args.stage == "simulate":
        simulate(args, config, project, env)
    elif args.stage == "train-predict":
        train_predict(args, config, project, env)
    else:
        bias_correct(args, config, project, env)


if __name__ == "__main__":
    main()
