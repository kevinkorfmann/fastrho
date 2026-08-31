#!/usr/bin/env python3
"""Prepare isolated ReLERNN sensitivity arms from the frozen bottleneck inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ARMS = {
    "bottleneck_constant_prior1_max256": {
        "history": "constant",
        "upper_rho_theta_ratio": 1.0,
        "max_sites": 256,
    },
    "bottleneck_matched_prior1_max256": {
        "history": "matched",
        "upper_rho_theta_ratio": 1.0,
        "max_sites": 256,
    },
    "bottleneck_matched_prior1_max1750": {
        "history": "matched",
        "upper_rho_theta_ratio": 1.0,
        "max_sites": 1750,
    },
}
SUFFIXES = {".vcf", ".npz", ".trees"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_inputs(source: Path) -> list[Path]:
    paths = [source / "genome.bed"]
    paths.extend(
        path
        for path in sorted(source.glob("region_*"))
        if path.is_file() and path.suffix in SUFFIXES
    )
    if not (source / "config.json").is_file() or len(paths) < 4:
        raise FileNotFoundError(f"Incomplete frozen input directory: {source}")
    return paths


def prepare(source_root: Path, target_root: Path, design: Path) -> dict:
    source = source_root / "validated_input" / "bottleneck_n20"
    files = frozen_inputs(source)
    source_config = json.loads((source / "config.json").read_text())
    target_root.mkdir(parents=True, exist_ok=True)
    arms_root = target_root / "arms"
    arms_root.mkdir(exist_ok=True)
    checksums = {path.name: sha256(path) for path in files}
    manifest = {
        "schema_version": 1,
        "source_root": str(source_root),
        "source": str(source),
        "source_checksums": checksums,
        "design_sha256": sha256(design),
        "arms": {},
    }
    for name, settings in ARMS.items():
        destination = arms_root / name
        if destination.exists():
            raise FileExistsError(f"Refusing to reuse sensitivity arm: {destination}")
        destination.mkdir()
        for path in files:
            os.link(path, destination / path.name)
        arm_config = dict(source_config)
        arm_config["name"] = name
        (destination / "config.json").write_text(
            json.dumps(arm_config, indent=2, sort_keys=True) + "\n"
        )
        manifest["arms"][name] = settings
    (target_root / "design.json").write_bytes(design.read_bytes())
    (target_root / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    args = parser.parse_args()
    manifest = prepare(
        args.source_root.resolve(), args.target_root.resolve(), args.design.resolve()
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
