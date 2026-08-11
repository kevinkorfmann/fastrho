"""Freeze one simulation-selected checkpoint per seed before Arabis inference."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_epoch(path: Path) -> int:
    match = re.search(r"epoch=(\d+)", path.name)
    if not match:
        raise ValueError(f"checkpoint lacks epoch: {path}")
    return int(match.group(1))


def validation_scores(root: Path) -> dict[int, float]:
    metrics_files = list((root / "logs").glob("**/metrics.csv"))
    if len(metrics_files) != 1:
        raise RuntimeError(f"expected one metrics.csv under {root}, found {metrics_files}")
    scores = {}
    with metrics_files[0].open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("val_pearson"):
                scores[int(row["epoch"])] = float(row["val_pearson"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-name", default="arabis_smalln_self3")
    parser.add_argument("--training-panel-sizes", default="8,12,16,24,25,32,50")
    parser.add_argument("--preregistration", type=Path, default=None)
    args = parser.parse_args()
    records = []
    reference_stats = None
    for seed in range(args.seeds):
        root = args.campaign / "seeds" / f"seed{seed}"
        checkpoints = list((root / "logs").glob("**/checkpoints/*.ckpt"))
        if not checkpoints:
            raise FileNotFoundError(f"no checkpoints for seed {seed}")
        scores = validation_scores(root)
        selected = max(checkpoints, key=lambda path: (scores[checkpoint_epoch(path)], path.name))
        stats = root / "shards" / "feat_stats.npz"
        stats_hash = sha256(stats)
        with np.load(stats) as archive:
            current_stats = {key: archive[key] for key in archive.files}
        if reference_stats is None:
            reference_stats = current_stats
        else:
            if current_stats.keys() != reference_stats.keys():
                raise RuntimeError(f"feature-stat keys differ for seed {seed}")
            for key in reference_stats:
                np.testing.assert_array_equal(reference_stats[key], current_stats[key])
        records.append({
            "seed": seed,
            "validation_pearson": scores[checkpoint_epoch(selected)],
            "checkpoint": str(selected.resolve()),
            "stats": str(stats.resolve()),
            "checkpoint_sha256": sha256(selected),
            "stats_sha256": stats_hash,
        })
    payload = {
        "schema_version": 1,
        "campaign": args.campaign_name,
        "selection_data": "simulated validation only",
        "arabis_cross_map_used_for_selection": False,
        "training_panel_sizes": [int(value) for value in args.training_panel_sizes.split(",")],
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "members": records,
    }
    if args.preregistration is not None:
        payload["preregistration"] = str(args.preregistration.resolve())
        payload["preregistration_sha256"] = sha256(args.preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output.read_text(), end="")


if __name__ == "__main__":
    main()
