#!/usr/bin/env python3
"""Prepare byte-identical evaluation arms for the demographic qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

SCENARIOS = ("bottleneck", "expansion")
METHODS = ("relernn", "pyrho")
HISTORIES = ("constant", "matched")
INPUT_SUFFIXES = (".vcf", ".npz", ".trees")
VALID_GT_ALLELES = {"0", "1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(source: Path) -> list[Path]:
    files = [source / "genome.bed"]
    files.extend(
        path
        for path in sorted(source.glob("region_*"))
        if path.is_file() and path.suffix in INPUT_SUFFIXES
    )
    missing = [path for path in files if not path.is_file()]
    if missing or len(files) < 4:
        raise FileNotFoundError(f"Incomplete frozen input directory: {source}")
    return files


def invalid_gt_reason(fields: list[str]) -> str | None:
    """Return why a VCF row is unusable, or None when every GT is biallelic."""
    formats = fields[8].split(":")
    if "GT" not in formats:
        return "missing_GT_format"
    gt_index = formats.index("GT")
    for sample in fields[9:]:
        values = sample.split(":")
        if gt_index >= len(values):
            return "missing_GT_value"
        alleles = re.split(r"[|/]", values[gt_index])
        if not alleles or any(allele not in VALID_GT_ALLELES for allele in alleles):
            return "non_biallelic_or_missing_GT"
    return None


def validate_vcf(source: Path, destination: Path) -> list[dict[str, str]]:
    """Copy a VCF while deterministically dropping rows pyrho cannot represent."""
    dropped = []
    with source.open() as input_handle, destination.open("w") as output_handle:
        for line in input_handle:
            if line.startswith("#"):
                output_handle.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                raise ValueError(f"Malformed VCF row in {source}: {line.rstrip()}")
            reason = invalid_gt_reason(fields)
            if reason is None:
                output_handle.write(line)
            else:
                dropped.append(
                    {
                        "file": source.name,
                        "chrom": fields[0],
                        "pos": fields[1],
                        "ref": fields[3],
                        "alt": fields[4],
                        "reason": reason,
                    }
                )
    return dropped


def prepare_validated_input(source: Path, destination: Path) -> tuple[list[Path], list[dict]]:
    """Create the outcome-blind validated input used by every paired arm."""
    destination.mkdir(parents=True, exist_ok=True)
    config_source = source / "config.json"
    config_target = destination / "config.json"
    if not config_target.exists():
        os.link(config_source, config_target)
    elif sha256(config_target) != sha256(config_source):
        raise RuntimeError(f"Validated config mismatch: {config_target}")

    dropped: list[dict] = []
    for path in source_files(source):
        target = destination / path.name
        if path.suffix == ".vcf":
            if target.exists():
                raise FileExistsError(f"Refusing to reuse validated VCF: {target}")
            records = validate_vcf(path, target)
            dropped.extend(records)
        elif not target.exists():
            os.link(path, target)
        elif sha256(target) != sha256(path):
            raise RuntimeError(f"Validated input mismatch: {target}")
    return source_files(destination), dropped


def arm_name(scenario: str, method: str, history: str) -> str:
    return f"{scenario}_{method}_{history}"


def prepare(root: Path) -> dict:
    input_root = root / "input"
    validated_root = root / "validated_input"
    arms_root = root / "arms"
    arms_root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema_version": 2,
        "vcf_validation_rule": (
            "Drop any data row before all method arms when any GT allele is not 0 or 1; "
            "retain raw frozen inputs and record every dropped coordinate."
        ),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        source = input_root / f"{scenario}_n20"
        config_path = source / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        config = json.loads(config_path.read_text())
        raw_files = source_files(source)
        raw_checksums = {path.name: sha256(path) for path in raw_files}
        validated_source = validated_root / f"{scenario}_n20"
        files, dropped = prepare_validated_input(source, validated_source)
        checksums = {path.name: sha256(path) for path in files}
        manifest["scenarios"][scenario] = {
            "raw_source": str(source.relative_to(root)),
            "validated_source": str(validated_source.relative_to(root)),
            "raw_checksums": raw_checksums,
            "validated_checksums": checksums,
            "dropped_vcf_records": dropped,
            "dropped_vcf_record_count": len(dropped),
            "arms": [],
        }

        for method in METHODS:
            for history in HISTORIES:
                name = arm_name(scenario, method, history)
                destination = arms_root / name
                destination.mkdir(parents=True, exist_ok=True)
                unexpected = [path for path in destination.iterdir() if path.name != "config.json"]
                if unexpected:
                    raise FileExistsError(f"Refusing to reuse nonempty arm: {destination}")

                for path in files:
                    target = destination / path.name
                    if not target.exists():
                        os.link(path, target)
                    if sha256(target) != checksums[path.name]:
                        raise RuntimeError(f"Input mismatch: {target}")

                arm_config = dict(config)
                arm_config["name"] = name
                if method == "pyrho" and history == "constant":
                    arm_config["popsizes"] = [arm_config["Ne"]]
                    arm_config["epochtimes"] = []
                (destination / "config.json").write_text(
                    json.dumps(arm_config, indent=2, sort_keys=True) + "\n"
                )
                manifest["scenarios"][scenario]["arms"].append(name)

    manifest_path = root / "input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(args.root.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
