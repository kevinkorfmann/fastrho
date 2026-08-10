#!/usr/bin/env python3
"""Diagnose paired constant- and matched-history benchmark predictions.

The benchmark summary reports pooled scores. This script returns to the exact,
unrounded prediction archive, keeps the preregistered joint-support rule, and
uses regions as bootstrap units when describing the matched-minus-constant
change in Pearson correlation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "paper" / "figdata" / "demography_matched_predictions.npz"
SUMMARY = ROOT / "paper" / "results_snapshot" / "demography_matched.json"
OUTPUT = ROOT / "paper" / "results_snapshot" / "demography_matched_diagnostics.json"
SCENARIOS = ("bottleneck", "expansion")
METHODS = ("relernn", "pyrho")
BOOTSTRAP_SEED = 20_260_722
BOOTSTRAP_REPLICATES = 10_000
FLOAT_DIGITS = 12


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_float(value: float) -> float:
    """Remove platform-level reduction noise from the committed JSON."""

    return round(float(value), FLOAT_DIGITS)


def _correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if kind == "pearson":
        return _stable_float(pearsonr(x, y).statistic)
    return _stable_float(spearmanr(x, y).statistic)


def _region_arrays(
    archive: np.lib.npyio.NpzFile, scenario: str, method: str
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    truth_prefix = f"truth__{scenario}__"
    suffixes = sorted(
        key.removeprefix(truth_prefix)
        for key in archive.files
        if key.startswith(truth_prefix)
    )
    regions = []
    for suffix in suffixes:
        truth = np.asarray(archive[f"truth__{scenario}__{suffix}"], dtype=float)
        constant = np.asarray(
            archive[f"pred__{scenario}_{method}_constant__{suffix}"], dtype=float
        )
        matched = np.asarray(
            archive[f"pred__{scenario}_{method}_matched__{suffix}"], dtype=float
        )
        if not (truth.shape == constant.shape == matched.shape):
            raise ValueError(f"unaligned arrays for {scenario}, {method}, {suffix}")
        keep = (
            np.isfinite(truth)
            & np.isfinite(constant)
            & np.isfinite(matched)
            & (truth > 0)
            & (constant > 0)
            & (matched > 0)
        )
        regions.append((truth[keep], constant[keep], matched[keep]))
    if not regions:
        raise ValueError(f"no archived regions for {scenario}, {method}")
    return regions


def _pearson_delta(regions: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    truth = np.concatenate([region[0] for region in regions])
    constant = np.concatenate([region[1] for region in regions])
    matched = np.concatenate([region[2] for region in regions])
    return _correlation(truth, matched, "pearson") - _correlation(
        truth, constant, "pearson"
    )


def _diagnose_arm_pair(
    regions: list[tuple[np.ndarray, np.ndarray, np.ndarray]], seed: int
) -> dict[str, object]:
    truth = np.concatenate([region[0] for region in regions])
    constant = np.concatenate([region[1] for region in regions])
    matched = np.concatenate([region[2] for region in regions])
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.integers(0, len(regions), size=len(regions))
        deltas[replicate] = _pearson_delta([regions[index] for index in sampled])

    constant_pearson = _correlation(truth, constant, "pearson")
    matched_pearson = _correlation(truth, matched, "pearson")
    constant_spearman = _correlation(truth, constant, "spearman")
    matched_spearman = _correlation(truth, matched, "spearman")
    return {
        "n_regions": len(regions),
        "n_windows": int(truth.size),
        "joint_support_rule": "finite, strictly positive truth and predictions in both arms",
        "constant": {
            "pearson": constant_pearson,
            "spearman": constant_spearman,
            "median_estimated_to_true": _stable_float(np.median(constant / truth)),
        },
        "matched": {
            "pearson": matched_pearson,
            "spearman": matched_spearman,
            "median_estimated_to_true": _stable_float(np.median(matched / truth)),
        },
        "matched_minus_constant": {
            "pearson": _stable_float(matched_pearson - constant_pearson),
            "spearman": _stable_float(matched_spearman - constant_spearman),
        },
        "prediction_agreement": {
            "pearson": _correlation(constant, matched, "pearson"),
            "spearman": _correlation(constant, matched, "spearman"),
            "median_matched_to_constant": _stable_float(np.median(matched / constant)),
        },
        "pearson_delta_region_bootstrap": {
            "seed": seed,
            "replicates": BOOTSTRAP_REPLICATES,
            "lower_95": _stable_float(np.quantile(deltas, 0.025)),
            "median": _stable_float(np.quantile(deltas, 0.5)),
            "upper_95": _stable_float(np.quantile(deltas, 0.975)),
        },
    }


def build() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "float_rounding_digits": FLOAT_DIGITS,
        "prediction_archive_sha256": _sha256(ARCHIVE),
        "benchmark_summary_sha256": _sha256(SUMMARY),
        "bootstrap_unit": "independent 2-Mb simulated region",
        "scenarios": {},
    }
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        for scenario_index, scenario in enumerate(SCENARIOS):
            scenario_payload = {}
            for method_index, method in enumerate(METHODS):
                seed = BOOTSTRAP_SEED + 10 * scenario_index + method_index
                scenario_payload[method] = _diagnose_arm_pair(
                    _region_arrays(archive, scenario, method), seed
                )
            payload["scenarios"][scenario] = scenario_payload
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
