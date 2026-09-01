#!/usr/bin/env python3
"""Stage isolated bottleneck/expansion constant-history competitor arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCENARIOS = ("bottleneck", "expansion")
# ``combined.vcf`` is intentionally excluded: run_relernn_paired.py regenerates it
# in place, so hard-linking it would mutate the source suite.  The per-region inputs
# and truth maps are immutable and safe to share by hard link.
INPUT_NAMES = ("genome.bed", "native_regime_generation.json")
INPUT_PATTERNS = ("region_*.vcf", "region_*.npz", "region_*.trees")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_verified(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"existing input differs: {destination}")
        return
    os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    arms = args.output / "arms"
    arms.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": "constant-demography misspecification at ReLERNN native windows",
        "source": str(args.source.resolve()),
        "scenarios": {},
        "settings": {
            "max_sites": 1750,
            "phased": True,
            "upper_rho_theta_ratio": 3.0,
            "seed": 1,
            "epochs": 100,
            "n_train": 100000,
            "n_validation": 10000,
            "n_test": 10000,
        },
    }
    for scenario in SCENARIOS:
        source = args.source / "arms" / scenario
        output = arms / scenario
        output.mkdir(parents=True, exist_ok=True)
        files = [source / name for name in INPUT_NAMES]
        for pattern in INPUT_PATTERNS:
            files.extend(sorted(source.glob(pattern)))
        if len(list(source.glob("region_*.vcf"))) != 20:
            raise ValueError(f"expected 20 VCFs in {source}")
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            link_verified(path, output / path.name)

        config = json.loads((source / "config.json").read_text())
        config["name"] = f"{scenario}_constant_history_native"
        config["popsizes"] = [float(config["Ne"])]
        config["epochtimes"] = []
        config_path = output / "config.json"
        encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
        if config_path.exists() and config_path.read_text() != encoded:
            raise ValueError(f"existing config differs: {config_path}")
        config_path.write_text(encoded)
        manifest["scenarios"][scenario] = {
            "config_sha256": sha256(config_path),
            "input_sha256": {path.name: sha256(path) for path in files},
        }

    design = args.output / "design.json"
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if design.exists() and design.read_text() != encoded:
        previous = json.loads(design.read_text())
        for scenario in SCENARIOS:
            prior_record = previous["scenarios"][scenario]
            next_record = manifest["scenarios"][scenario]
            if prior_record["config_sha256"] != next_record["config_sha256"]:
                raise ValueError(f"existing config provenance differs: {design}")
            for name, checksum in prior_record["input_sha256"].items():
                if next_record["input_sha256"].get(name) != checksum:
                    raise ValueError(f"existing input provenance differs: {name}")
    design.write_text(encoded)
    print(design)


if __name__ == "__main__":
    main()
