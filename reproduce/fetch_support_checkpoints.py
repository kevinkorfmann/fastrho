#!/usr/bin/env python3
"""Download and verify one frozen paper-only checkpoint group."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "checkpoints.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    groups = {record["id"]: record for record in registry["groups"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=sorted(groups))
    parser.add_argument("--output-dir", type=Path, default=Path("paper-checkpoints"))
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve() / args.group
    output.mkdir(parents=True, exist_ok=True)
    base = registry["paper_support_release"].replace("/tag/", "/download/")
    for filename, expected in groups[args.group]["files"].items():
        target = output / filename
        if not target.is_file() or sha256(target) != expected:
            temporary = target.with_suffix(target.suffix + ".part")
            request = urllib.request.Request(
                f"{base}/{filename}", headers={"User-Agent": "fastrho-paper-reproduction"}
            )
            with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
            os.replace(temporary, target)
        if sha256(target) != expected:
            raise RuntimeError(f"SHA-256 mismatch: {filename}")
        print(target)
    print(f"verified {len(groups[args.group]['files'])} files for {args.group}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
