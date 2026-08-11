"""Re-derive the reported gene-conversion stress-test endpoints from raw arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manuscript_source import MAIN, SI

ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "paper" / "results_snapshot" / "gene_conversion.json"
ARRAY_PATH = ROOT / "paper" / "figdata" / "gene_conversion.npz"


def _condition_index(condition_ids: np.ndarray, condition_id: str) -> int:
    matches = np.flatnonzero(condition_ids == condition_id)
    assert matches.size == 1
    return int(matches[0])


def test_gene_conversion_design_is_complete_and_paired() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    design = summary["design"]
    assert design["n_regions"] == 24
    assert design["map_kinds"] == {"gp": 12, "hotspot": 12}
    assert design["gene_conversion_to_mean_crossover_ratios"] == [0.0, 0.5, 1.0, 2.0, 4.0]
    assert design["gene_conversion_tract_lengths"] == [100, 300, 1000]
    assert design["paired_crossover_maps_across_conditions"] is True
    assert design["checkpoint_retrained"] is False
    assert design["n_bootstrap"] == 2_000
    assert len(summary["conditions"]) == 13


def test_gene_conversion_endpoint_metrics_rederive_from_raw_arrays() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    conditions = {row["id"]: row for row in summary["conditions"]}
    arrays = np.load(ARRAY_PATH, allow_pickle=False)
    truth = arrays["truth_25kb"]
    predicted = arrays["predicted_25kb"]
    ids = arrays["condition_id"]

    assert truth.shape == (24, 40)
    assert predicted.shape == (13, 24, 40)
    assert np.array_equal(ids, np.asarray([row["id"] for row in summary["conditions"]]))

    baseline_index = _condition_index(ids, "gc0")
    for condition_id in ("gc0", "gc4_t100", "gc4_t300", "gc4_t1000"):
        index = _condition_index(ids, condition_id)
        observed_pearson = float(np.corrcoef(predicted[index].ravel(), truth.ravel())[0, 1])
        observed_scale = float(
            np.median(
                np.mean(predicted[index], axis=1)
                / np.mean(predicted[baseline_index], axis=1)
            )
        )
        reported = conditions[condition_id]["pooled"]
        assert np.isclose(observed_pearson, reported["pearson"]["estimate"], atol=1e-12)
        assert np.isclose(
            observed_scale,
            reported["paired_mean_rate_vs_no_conversion"]["estimate"],
            atol=1e-12,
        )


def test_gene_conversion_claims_match_committed_results() -> None:
    assert "$r=0.873$" in MAIN
    assert "correlation was $r=0.870$ for 100-bp mean tracts" in MAIN
    assert "$r=0.678$ for 1,000-bp mean tracts" in MAIN
    assert "2.89-fold higher" in MAIN
    assert "fixed model" in SI
    assert "2,000 bootstrap resamples" in SI
    assert "crossover map" in SI


def test_gene_conversion_display_replicate_is_deterministic() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    conditions = {row["id"]: row for row in summary["conditions"]}
    arrays = np.load(ARRAY_PATH, allow_pickle=False)
    truth = arrays["truth_25kb"]
    predicted = arrays["predicted_25kb"]
    ids = arrays["condition_id"]
    baseline = predicted[_condition_index(ids, "gc0")]
    severe = predicted[_condition_index(ids, "gc4_t1000")]

    def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a - np.mean(a, axis=1, keepdims=True)
        b = b - np.mean(b, axis=1, keepdims=True)
        return np.sum(a * b, axis=1) / np.sqrt(
            np.sum(a**2, axis=1) * np.sum(b**2, axis=1)
        )

    metrics = np.column_stack(
        [
            rowwise_pearson(truth, baseline),
            rowwise_pearson(truth, severe),
            np.mean(severe, axis=1) / np.mean(baseline, axis=1),
        ]
    )
    target = np.asarray(
        [
            conditions["gc0"]["pooled"]["pearson"]["estimate"],
            conditions["gc4_t1000"]["pooled"]["pearson"]["estimate"],
            conditions["gc4_t1000"]["pooled"][
                "paired_mean_rate_vs_no_conversion"
            ]["estimate"],
        ]
    )
    median = np.median(metrics, axis=0)
    mad = np.median(np.abs(metrics - median), axis=0)
    score = np.sum(np.abs(metrics - target) / mad, axis=1)

    assert int(np.argmin(score)) == 22
    assert np.allclose(metrics[22], [0.88247808, 0.66212324, 2.34870777])
    assert "Each curve is divided by its regional mean" in SI
