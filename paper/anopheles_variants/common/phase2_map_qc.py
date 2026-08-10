#!/usr/bin/env python3
"""Validate every promoted Phase 2 map before downstream analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ARMS = ("2R", "2L", "3R", "3L", "X")


def scalar(value):
    return value.item() if isinstance(value, np.ndarray) else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.selection.open(newline="", encoding="utf-8") as handle:
        cohorts = [row["cohort"] for row in csv.DictReader(handle, delimiter="\t")]
    expected = {f"{cohort}__{arm}" for cohort in cohorts for arm in ARMS}
    observed = {path.stem for path in args.maps.glob("*.npz")}
    errors: list[str] = []
    if observed != expected:
        errors.append(f"map identity mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    rows = []
    required = {
        "pop", "region", "n_hap", "n_snp", "Ne_est",
        "starts_50000", "rho_50000", "r_50000",
        "starts_100000", "rho_100000", "r_100000",
    }
    for stem in sorted(expected & observed):
        path = args.maps / f"{stem}.npz"
        with np.load(path, allow_pickle=True) as data:
            missing = sorted(required - set(data.files))
            if missing:
                errors.append(f"{stem}: missing arrays {missing}")
                continue
            cohort, arm = stem.split("__", 1)
            if str(scalar(data["pop"])) != cohort or str(scalar(data["region"])) != arm:
                errors.append(f"{stem}: embedded cohort/arm mismatch")
            n_hap = int(scalar(data["n_hap"]))
            n_snp = int(scalar(data["n_snp"]))
            ne = float(scalar(data["Ne_est"]))
            if n_hap != 80 or n_snp < 2 or not np.isfinite(ne) or ne <= 0:
                errors.append(f"{stem}: invalid n_hap={n_hap}, n_snp={n_snp}, Ne={ne}")
            window_rows = {}
            for window in (50_000, 100_000):
                starts = np.asarray(data[f"starts_{window}"], dtype=float)
                rho = np.asarray(data[f"rho_{window}"], dtype=float)
                rate = np.asarray(data[f"r_{window}"], dtype=float)
                if not (len(starts) == len(rho) == len(rate) and len(starts) > 0):
                    errors.append(f"{stem}: inconsistent {window}-bp array lengths")
                    continue
                if np.any(np.diff(starts) <= 0):
                    errors.append(f"{stem}: non-increasing {window}-bp coordinates")
                finite_fraction = float(np.mean(np.isfinite(rate)))
                positive_fraction = float(np.mean(np.isfinite(rate) & (rate > 0)))
                if finite_fraction < 0.99 or positive_fraction < 0.90:
                    errors.append(
                        f"{stem}: poor {window}-bp rate coverage "
                        f"(finite={finite_fraction:.3f}, positive={positive_fraction:.3f})"
                    )
                window_rows[str(window)] = {
                    "n_windows": len(starts),
                    "finite_rate_fraction": finite_fraction,
                    "positive_rate_fraction": positive_fraction,
                }
        summary_path = path.with_suffix(".json")
        if not summary_path.is_file():
            errors.append(f"{stem}: missing JSON sidecar")
        else:
            summary = json.loads(summary_path.read_text())
            if summary.get("cohort") != cohort or summary.get("arm") != arm:
                errors.append(f"{stem}: JSON sidecar identity mismatch")
            if int(summary.get("n_hap", -1)) != n_hap or int(summary.get("n_snp", -1)) != n_snp:
                errors.append(f"{stem}: JSON/NPZ count mismatch")
        rows.append({
            "cohort": cohort,
            "arm": arm,
            "n_hap": n_hap,
            "n_snp": n_snp,
            "Ne_est": ne,
            "windows": window_rows,
        })
    report = {
        "schema_version": 1,
        "variant": "phase2",
        "expected_maps": len(expected),
        "observed_maps": len(observed),
        "status": "failed" if errors else "passed",
        "errors": errors,
        "maps": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"verified {len(rows)} Phase 2 maps")


if __name__ == "__main__":
    main()
