#!/usr/bin/env python3
"""Select a training checkpoint from exact Lightning validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

try:  # package import in tests
    from scripts.package_model_release import sha256
except ModuleNotFoundError:  # direct ``python scripts/select_model_checkpoint.py`` invocation
    from package_model_release import sha256


def metric_rows(path: Path, metric: str) -> list[tuple[float, int]]:
    rows: list[tuple[float, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if metric not in (reader.fieldnames or ()) or "epoch" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} lacks {metric!r} or 'epoch'")
        for row in reader:
            raw_metric = row.get(metric, "")
            raw_epoch = row.get("epoch", "")
            if raw_metric in (None, "") or raw_epoch in (None, ""):
                continue
            value = float(raw_metric)
            epoch_float = float(raw_epoch)
            if not math.isfinite(value) or not epoch_float.is_integer():
                continue
            rows.append((value, int(epoch_float)))
    if not rows:
        raise ValueError(f"{path} contains no finite {metric!r} validation rows")
    return rows


def select(rows: list[tuple[float, int]], mode: str) -> tuple[float, int]:
    # Earliest epoch is the deterministic tie-breaker.
    return (max(rows, key=lambda item: (item[0], -item[1])) if mode == "max"
            else min(rows, key=lambda item: (item[0], item[1])))


def matching_checkpoint(directory: Path, epoch: int) -> Path:
    matches = sorted(directory.glob(f"epoch={epoch}-*.ckpt"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one epoch={epoch} checkpoint in {directory}, found {len(matches)}"
        )
    return matches[0]


def git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--metric", default="val_pearson")
    parser.add_argument("--mode", choices=("max", "min"), default="max")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    value, epoch = select(metric_rows(args.metrics, args.metric), args.mode)
    checkpoint = matching_checkpoint(args.checkpoint_dir, epoch)
    root = Path(__file__).resolve().parents[1]
    record = {
        "schema_version": 1,
        "model_id": args.model_id,
        "selection": {
            "metric": args.metric,
            "mode": args.mode,
            "tie_breaker": "earliest_epoch",
            "epoch": epoch,
            "value": value,
        },
        "source": {
            "repository_revision": git_revision(root),
            "metrics_name": args.metrics.name,
            "metrics_sha256": sha256(args.metrics),
        },
        "files": {
            "checkpoint": {
                "name": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256(checkpoint),
            },
            "stats": {
                "name": args.stats.name,
                "bytes": args.stats.stat().st_size,
                "sha256": sha256(args.stats),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={checkpoint}")
    print(f"selection_record={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
