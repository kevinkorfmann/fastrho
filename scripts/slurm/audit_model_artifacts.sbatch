#!/bin/bash
#SBATCH --job-name=fastrho-model-audit
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
#SBATCH --output=slurm-model-audit-%j.out

set -euo pipefail

if (( $# < 1 )); then
    echo "usage: sbatch $0 MANIFEST.json [MANIFEST.json ...]" >&2
    exit 2
fi

python3 - "$@" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifacts(value):
    if isinstance(value, dict):
        for path_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("stats", "stats_sha256"),
            ("preregistration", "preregistration_sha256"),
        ):
            path = value.get(path_key)
            expected = value.get(hash_key)
            if path and expected and not str(expected).startswith("ensemble_manifest:"):
                yield path_key, Path(path), str(expected)
        for child in value.values():
            yield from artifacts(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifacts(child)


rows = {}
for manifest_name in sys.argv[1:]:
    manifest = Path(manifest_name)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for kind, path, expected in artifacts(payload):
        rows[(str(path), expected)] = (manifest, kind, path, expected)

failures = 0
print("status\tkind\tbytes\tsha256\tmanifest\tpath")
for manifest, kind, path, expected in sorted(rows.values(), key=lambda row: str(row[2])):
    if not path.is_file():
        print(f"missing\t{kind}\t0\t{expected}\t{manifest}\t{path}")
        failures += 1
        continue
    observed = sha256(path)
    status = "ok" if observed == expected else "mismatch"
    print(f"{status}\t{kind}\t{path.stat().st_size}\t{observed}\t{manifest}\t{path}")
    failures += observed != expected

if failures:
    raise SystemExit(f"{failures} model artifact checks failed")
PY
