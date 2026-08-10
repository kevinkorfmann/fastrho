"""Evaluate predeclared Arabis resamples against full maps and the F2 map."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from arabis_cross_eval import CHROMS, binned_prediction, prediction_span, read_cross_map, relative


def compare(
    panel_dir: Path, panel_stem: str, full_dir: Path, full_stem: str, cross, width: int
) -> dict:
    subset_values, full_values, cross_values = [], [], []
    per_chrom = {}
    for chrom in CHROMS:
        subset_path = panel_dir / f"{panel_stem}.{chrom}.npz"
        full_path = full_dir / f"{full_stem}.{chrom}.npz"
        bp, cm = cross[chrom]
        spans = [prediction_span(subset_path), prediction_span(full_path)]
        start = np.ceil(max(bp[0], *(x[0] for x in spans)) / width) * width
        stop = np.floor(min(bp[-1], *(x[1] for x in spans)) / width) * width
        edges = np.arange(start, stop + width, width, dtype=float)
        sub = relative(binned_prediction(subset_path, edges))
        full = relative(binned_prediction(full_path, edges))
        observed = relative(np.diff(np.interp(edges, bp, cm)) / (width / 1e6))
        subset_values.append(sub)
        full_values.append(full)
        cross_values.append(observed)
        per_chrom[chrom] = {
            "subset_vs_full": float(spearmanr(sub, full).statistic),
            "subset_vs_cross": float(spearmanr(sub, observed).statistic),
        }
    subset = np.concatenate(subset_values)
    full = np.concatenate(full_values)
    observed = np.concatenate(cross_values)
    return {
        "n_windows": len(subset),
        "spearman_subset_vs_full": float(spearmanr(subset, full).statistic),
        "spearman_subset_vs_cross": float(spearmanr(subset, observed).statistic),
        "per_chromosome": per_chrom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--full-maps", type=Path, required=True)
    parser.add_argument("--cross-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-bp", type=int, default=2_000_000)
    args = parser.parse_args()
    cross, audit = read_cross_map(args.cross_map)
    rows = list(csv.DictReader(args.manifest.open(newline=""), delimiter="\t"))
    results = []
    for row in rows:
        result = dict(row)
        result.update(compare(
            args.maps_root / row["panel"] / "ensemble", row["panel"], args.full_maps,
            row["species"], cross, args.window_bp,
        ))
        results.append(result)
    by_design = {}
    for design in sorted({r["design"] for r in results}):
        chosen = [r for r in results if r["design"] == design]
        by_design[design] = {
            key: {
                "median": float(np.median([r[key] for r in chosen])),
                "range": [float(np.min([r[key] for r in chosen])),
                          float(np.max([r[key] for r in chosen]))],
            } for key in ("spearman_subset_vs_full", "spearman_subset_vs_cross")
        }
    output = {
        "design_note": "Subsets and frozen models were selected without the F2 map.",
        "window_mb": args.window_bp / 1e6,
        "cross_map_audit": audit,
        "by_design": by_design,
        "panels": results,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(by_design, indent=2))


if __name__ == "__main__":
    main()
