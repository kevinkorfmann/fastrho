#!/usr/bin/env python3
"""Validate and record one runtime used by the paired demography benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

PYRHO_VERSIONS = {
    "pyrho": "0.2.1",
    "ldpop": "1.0.2",
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "msprime": "1.3.4",
    "numba": "0.60.0",
    "llvmlite": "0.43.0",
    "pandas": "2.1.4",
    "tables": "3.11.1",
    "cyvcf2": "0.31.4",
}

RELERNN_PACKAGES = (
    "ReLERNN",
    "tensorflow",
    "msprime",
    "numpy",
    "scipy",
    "scikit-learn",
    "scikit-allel",
    "dask",
    "h5py",
    "matplotlib",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pyrho_record() -> dict:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Expected Python 3.12, observed {sys.version}")
    packages = {package: version(package) for package in PYRHO_VERSIONS}
    if packages != PYRHO_VERSIONS:
        raise RuntimeError(f"pyrho runtime mismatch: {packages}")
    return {"python": sys.version, "packages": packages}


def relernn_record(image: Path, patch: Path) -> dict:
    packages = {package: version(package) for package in RELERNN_PACKAGES}
    if packages["ReLERNN"] != "2.0.0":
        raise RuntimeError(f"Unexpected ReLERNN version: {packages['ReLERNN']}")
    return {
        "image": "nvcr.io/nvidia/tensorflow:25.02-tf2-py3",
        "image_sha256": sha256(image),
        "relernn_commit": "6655efd",
        "compatibility_patch": patch.name,
        "compatibility_patch_sha256": sha256(patch),
        "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pyrho", "relernn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--patch", type=Path)
    args = parser.parse_args()
    if args.mode == "pyrho":
        record = pyrho_record()
    else:
        if args.image is None or args.patch is None:
            parser.error("--image and --patch are required for --mode relernn")
        record = relernn_record(args.image, args.patch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
