#!/usr/bin/env python3
"""Validate the frozen ReLERNN intended-regime reviewer-response suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


SCENARIOS = ("constant", "bottleneck", "expansion", "decode", "hapmap", "dog")
MU = {scenario: 1.5e-8 for scenario in SCENARIOS}
MU["dog"] = 4e-9
UPPER = {scenario: 3.0 for scenario in SCENARIOS}
UPPER["dog"] = 8.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intended-provenance", required=True, type=Path)
    parser.add_argument("--source-provenance", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--raw-results", required=True, type=Path)
    parser.add_argument("--bscorrect-results", required=True, type=Path)
    parser.add_argument("--misspecified-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    design = json.loads(args.design.read_text())
    assert design["analysis_status"] == "prespecified_before_generation"
    assert "do not tune or select" in design["decision_rule"].lower()
    inherited = design["training_settings_inherited"]
    assert inherited["max_sites"] == 1750
    assert inherited["phased"] is True
    assert inherited["epochs"] == 100
    assert inherited["upper_rho_theta_ratio"] == {"dog": 8, "human": 3}

    raw = json.loads(args.raw_results.read_text())
    bscorrect = json.loads(args.bscorrect_results.read_text())
    misspecified = json.loads(args.misspecified_results.read_text())
    report = {
        "schema_version": 1,
        "status": "valid",
        "decision_rule": design["decision_rule"],
        "raw_results_sha256": sha256(args.raw_results),
        "bscorrect_results_sha256": sha256(args.bscorrect_results),
        "misspecified_results_sha256": sha256(args.misspecified_results),
        "scenarios": {},
        "misspecified_controls": {},
    }

    for scenario in SCENARIOS:
        arm = args.intended_provenance / "data" / "arms" / scenario
        source = args.source_provenance / "arms" / scenario
        generation = json.loads((arm / "native_regime_generation.json").read_text())
        reuse = json.loads((arm / "relernn_reuse_manifest.json").read_text())
        run = json.loads((source / "relernn_run_manifest.json").read_text())
        score = raw["scenarios"][scenario]
        corrected_score = bscorrect["scenarios"][scenario]

        assert generation["scenario"] == scenario
        assert generation["mutation_rate"] == MU[scenario]
        assert generation["sequence_length_bp"] == 10_000_000
        assert generation["n_diploid"] == 10
        assert len(generation["regions"]) == 20
        for index, record in enumerate(generation["regions"]):
            assert record["region"] == index
            truth = arm / f"region_{index:03d}.npz"
            assert sha256(truth) == record["truth_sha256"]
            with np.load(truth) as data:
                rates = np.asarray(data["map_rate"], dtype=float)
            assert rates.size > 0 and np.all(np.isfinite(rates))
            assert np.allclose(rates, record["constant_rate_per_bp"], rtol=0, atol=0)

        command = reuse["prediction_command"]
        assert "--phased" in command
        assert command[command.index("--seed") + 1] == "1"
        assert reuse["operation"] == "frozen_model_reuse_prediction"
        assert reuse["source_model_capacity"] >= reuse["max_input_sites"]
        prediction = arm / "relernn_project" / "combined.PREDICT.txt"
        assert sha256(prediction) == reuse["prediction_sha256"]

        assert run["max_sites"] == 1750
        assert run["upper_rho_theta_ratio"] == UPPER[scenario]
        assert run["n_train"] == 100_000
        assert run["n_validation"] == 10_000
        assert run["n_test"] == 10_000
        assert run["epochs"] == 100
        simulate = run["simulate_command"]
        assert simulate[simulate.index("--assumedMu") + 1] == str(MU[scenario])
        assert "--phased" in simulate

        test_results = source / "relernn_project" / "networks" / "testResults.p"
        window_sizes = source / "relernn_project" / "networks" / "windowSizes.txt"
        assert sha256(test_results) == reuse["source_test_results_sha256"]
        assert sha256(window_sizes) == reuse["source_window_sizes_sha256"]
        heldout = pickle.loads(test_results.read_bytes())
        heldout_r = float(
            np.corrcoef(
                np.ravel(heldout["predictions"]),
                np.ravel(heldout["Y_test"]),
            )[0, 1]
        )
        assert heldout_r >= 0.90

        assert score["complete_windows_only"] is True
        assert score["n_regions"] == 20
        assert score["n_missing_complete_windows"] == 0
        assert score["n_retained_windows"] > 0
        assert score["native_truth_rate"]["q999_rho_theta_ratio"] <= UPPER[scenario]
        for method in ("fastrho", "pyrho", "relernn"):
            metrics = score["methods"][method]
            assert metrics["n"] == score["n_retained_windows"]
            assert metrics["n_nonfinite_predictions"] == 0
            expected_zeros = 9 if (scenario, method) == ("decode", "relernn") else 0
            assert metrics["n_zero_predictions"] == expected_zeros
            assert math.isfinite(metrics["pearson"])

        assert corrected_score["complete_windows_only"] is True
        assert corrected_score["n_retained_windows"] == score["n_retained_windows"]
        assert corrected_score["checksums"]["truth_sha256"] == score["checksums"]["truth_sha256"]
        corrected_prediction = arm / "relernn_project" / "combined.PREDICT.BSCORRECTED.txt"
        assert sha256(corrected_prediction) == corrected_score["checksums"]["relernn_prediction_sha256"]
        corrected_metrics = corrected_score["methods"]["relernn"]
        assert corrected_metrics["n"] == score["n_retained_windows"]
        assert corrected_metrics["n_nonfinite_predictions"] == 0
        expected_corrected_zeros = 9 if scenario == "decode" else 0
        assert corrected_metrics["n_zero_predictions"] == expected_corrected_zeros
        assert math.isfinite(corrected_metrics["pearson"])

        report["scenarios"][scenario] = {
            "heldout_relernn_pearson": heldout_r,
            "native_window_median_bp": score["native_window_bp"]["median"],
            "complete_windows": score["n_retained_windows"],
            "maximum_truth_rho_theta_ratio": score["native_truth_rate"]["q999_rho_theta_ratio"],
            "upper_rho_theta_ratio": UPPER[scenario],
            "pearson": {
                method: score["methods"][method]["pearson"]
                for method in ("fastrho", "pyrho", "relernn")
            },
            "bscorrect_relernn_pearson": corrected_metrics["pearson"],
            "zero_predictions": {
                method: score["methods"][method]["n_zero_predictions"]
                for method in ("fastrho", "pyrho", "relernn")
            },
        }

    for scenario in ("bottleneck", "expansion"):
        matched = raw["scenarios"][scenario]
        constant = misspecified["scenarios"][scenario]
        assert matched["checksums"]["truth_sha256"] == constant["checksums"]["truth_sha256"]
        assert matched["n_retained_windows"] == constant["n_retained_windows"]
        assert constant["n_missing_complete_windows"] == 0
        for method in ("pyrho", "relernn"):
            metrics = constant["methods"][method]
            assert metrics["n_nonfinite_predictions"] == 0
            assert metrics["n_zero_predictions"] == 0
        report["misspecified_controls"][scenario] = {
            "truth_checksums_identical": True,
            "complete_windows": constant["n_retained_windows"],
            "pearson": {
                method: constant["methods"][method]["pearson"]
                for method in ("pyrho", "relernn")
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
