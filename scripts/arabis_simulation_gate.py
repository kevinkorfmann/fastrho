"""Apply a preregistered, simulation-only gate before Arabis inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    prereg = json.loads(args.preregistration.read_text())
    gates = prereg["gates"]
    failures = []
    validation = [float(member["validation_pearson"]) for member in manifest["members"]]
    minimum_validation = float(gates["minimum_member_validation_pearson"])
    if min(validation) < minimum_validation:
        failures.append(
            f"minimum member validation Pearson {min(validation):.4f} < {minimum_validation:.4f}"
        )

    audits = [
        json.loads((args.audit_dir / f"seed{index}.json").read_text())
        for index in range(len(manifest["members"]))
    ]
    stratum_summary = {}
    for stratum in sorted(gates["minimum_median_true_Ne_100kb_pearson"]):
        row = {}
        for mode, threshold_key in (
            ("true", "minimum_median_true_Ne_100kb_pearson"),
            ("estimated", "minimum_median_estimated_Ne_100kb_pearson"),
        ):
            metrics = [audit["strata"][stratum][mode] for audit in audits]
            correlations = [float(item["100kb"]["pearson"]) for item in metrics]
            shard_counts = [int(item["n_shards"]) for item in metrics]
            threshold = float(gates[threshold_key][stratum])
            median = float(np.median(correlations))
            row[mode] = {
                "member_100kb_pearson": correlations,
                "median_100kb_pearson": median,
                "threshold": threshold,
                "n_shards": shard_counts,
            }
            if median < threshold:
                failures.append(
                    f"{stratum} {mode}-Ne median 100kb Pearson {median:.4f} < {threshold:.4f}"
                )
            required_n = int(gates["minimum_shards_per_audit_stratum"])
            if min(shard_counts) < required_n:
                failures.append(
                    f"{stratum} minimum evaluated shards {min(shard_counts)} < {required_n}"
                )
        stratum_summary[stratum] = row

    payload = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "decision_data": "simulated validation and held-out simulated audit only",
        "arabis_cross_map_used": False,
        "member_validation_pearson": validation,
        "minimum_member_validation_pearson": minimum_validation,
        "strata": stratum_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("simulation-only gate failed; Arabis inference is blocked")


if __name__ == "__main__":
    main()
