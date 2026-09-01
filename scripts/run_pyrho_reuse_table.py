#!/usr/bin/env python3
"""Run pyrho optimize with a frozen, demography-matched lookup table."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_dir", type=Path)
    parser.add_argument("--table-file", required=True, type=Path)
    parser.add_argument("--numthreads", type=int, default=8)
    args = parser.parse_args()
    config_dir = args.config_dir.resolve()
    table = args.table_file.resolve()
    if not table.is_file():
        raise FileNotFoundError(table)
    pyrho = Path(sys.executable).with_name("pyrho")
    if not pyrho.is_file():
        raise FileNotFoundError(pyrho)
    region_wall = {}
    optimized = []
    started = time.perf_counter()
    for vcf_text in sorted(glob.glob(str(config_dir / "region_*.vcf"))):
        vcf = Path(vcf_text)
        output = vcf.with_suffix(".rmap")
        if output.exists():
            raise FileExistsError(output)
        region_started = time.perf_counter()
        command = [
            str(pyrho),
            "optimize",
            "--vcffile",
            str(vcf),
            "--tablefile",
            str(table),
            "--ploidy",
            "1",
            "-w",
            "50",
            "-bpen",
            "25",
            "--numthreads",
            str(args.numthreads),
            "-o",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        elapsed = float(time.perf_counter() - region_started)
        region_wall[vcf.stem] = elapsed
        if result.returncode != 0:
            raise RuntimeError(
                f"pyrho failed for {vcf.name}:\n{result.stdout}\n{result.stderr}"
            )
        optimized.append(
            {
                "region": vcf.stem,
                "vcf_sha256": sha256(vcf),
                "rmap_sha256": sha256(output),
                "wall_seconds": elapsed,
            }
        )
    total = float(time.perf_counter() - started)
    if len(optimized) != 20:
        raise ValueError(f"Expected 20 optimized regions, found {len(optimized)}")
    manifest = {
        "schema_version": 1,
        "lookup_table_reused": True,
        "lookup_table": str(table),
        "lookup_table_sha256": sha256(table),
        "lookup_table_wall_seconds": 0.0,
        "optimize_wall_seconds": total,
        "total_wall_seconds": total,
        "region_wall_seconds": region_wall,
        "optimized": optimized,
    }
    (config_dir / "pyrho_runtime.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"pyrho: {len(optimized)} regions in {total:.1f}s")


if __name__ == "__main__":
    main()
