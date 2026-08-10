"""Bound generic sweep-induced cold-rate artefacts in fastrho.

This is a conservative companion to the Anopheles resistance-locus controls.
It asks a narrow question using the committed SLiM dose-response arrays where
the true recombination map is known: under hard sweeps, how often does fastrho
underestimate the true rate by at least as much as the resistance-locus
cold-spot ratio?  It does not replace allele-defined resistance-carrier or
sweep-core controls.

Writes paper/figdata/selection_artifact_bound.json.
Run: python scripts/selection_artifact_bound.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(HERE, "paper", "figdata", "selection_dr_figdata.npz")
OUT = os.path.join(HERE, "paper", "figdata", "selection_artifact_bound.json")

THRESHOLDS = (0.25, 0.35, 0.50)


def _summary(true: np.ndarray, pred: np.ndarray) -> dict:
    m = np.isfinite(true) & np.isfinite(pred) & (true > 0) & (pred > 0)
    ratio = pred[m] / true[m]
    return {
        "n_windows": int(ratio.size),
        "median_pred_true": float(np.median(ratio)),
        "q05_pred_true": float(np.percentile(ratio, 5)),
        "q95_pred_true": float(np.percentile(ratio, 95)),
        "frac_pred_true_le": {str(t): float(np.mean(ratio <= t)) for t in THRESHOLDS},
    }


def main() -> None:
    z = np.load(IN)
    result = {
        "metadata": {
            "source_npz": "paper/figdata/selection_dr_figdata.npz",
            "interpretation": (
                "Calibration-bound test for generic sweep artefacts. "
                "It uses simulated hard sweeps with known true maps and does "
                "not replace allele-defined resistance-carrier or sweep-core "
                "controls at real resistance loci."
            ),
            "thresholds_pred_true": list(THRESHOLDS),
        },
        "neutral": {
            "fastrho": _summary(z["calib_true_neutral"], z["calib_fastrho_neutral"]),
            "pyrho": _summary(z["calib_true_neutral"], z["calib_pyrho_neutral"]),
        },
        "hard_sweep": {
            "fastrho": _summary(z["calib_true_sweep"], z["calib_fastrho_sweep"]),
            "pyrho": _summary(z["calib_true_sweep"], z["calib_pyrho_sweep"]),
        },
    }
    for method in ("fastrho", "pyrho"):
        neutral = result["neutral"][method]["frac_pred_true_le"]
        sweep = result["hard_sweep"][method]["frac_pred_true_le"]
        result["hard_sweep"][method]["excess_frac_vs_neutral"] = {
            t: float(sweep[t] - neutral[t]) for t in sweep
        }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)
    print("wrote", OUT)
    for method in ("fastrho", "pyrho"):
        sw = result["hard_sweep"][method]
        ex = sw["excess_frac_vs_neutral"]["0.35"]
        print("%s hard-sweep median pred/true %.2fx; frac <=0.35 %.1f%% (excess %.1f pp)" % (
            method, sw["median_pred_true"],
            100 * sw["frac_pred_true_le"]["0.35"], 100 * ex))


if __name__ == "__main__":
    main()
