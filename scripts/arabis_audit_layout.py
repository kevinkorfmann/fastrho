"""Stratify held-out simulation shards using generator metadata only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("ts_*.npz"))
    if len(paths) != args.expected:
        raise RuntimeError(f"audit shards={len(paths)}, expected {args.expected}")
    counts = Counter()
    for path in paths:
        with np.load(path, allow_pickle=True) as archive:
            meta = json.loads(str(archive["meta"]))
        design = meta["design"]
        destination = args.output_root / design
        destination.mkdir(parents=True, exist_ok=True)
        link = destination / path.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(path.resolve())
        counts[design] += 1
    payload = {"expected": args.expected, "strata": dict(sorted(counts.items()))}
    (args.output_root / "layout.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
