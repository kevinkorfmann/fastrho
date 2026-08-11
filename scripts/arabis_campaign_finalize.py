"""Verify and checksum a completed structured-selfing Arabis campaign."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign
    required = {
        name: campaign / filename
        for name, filename in {
            "preregistration": "preregistration.json",
            "preregistered_inputs": "preregistered_inputs.sha256",
            "smoke": "smoke/smoke.json",
            "dataset": "dataset.tsv",
            "simulation_design_audit": "simulation_design_audit.json",
            "frozen_ensemble": "frozen_ensemble.json",
            "simulation_gate": "simulation_gate.json",
            "cross_results": "arabis_cross_results.json",
            "cross_windows": "arabis_cross_windows.npz",
        }.items()
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required campaign artifacts are missing: {missing}")

    prereg = json.loads(required["preregistration"].read_text())
    frozen = json.loads(required["frozen_ensemble"].read_text())
    gate = json.loads(required["simulation_gate"].read_text())
    result = json.loads(required["cross_results"].read_text())
    design = json.loads(required["simulation_design_audit"].read_text())
    assert prereg["campaign"] == "arabis_structured_selfing_v1"
    assert frozen["campaign"] == prereg["campaign"]
    assert frozen["arabis_cross_map_used_for_selection"] is False
    assert len(frozen["members"]) == prereg["ensemble_members"] == 7
    assert gate["passed"] is True and gate["arabis_cross_map_used"] is False
    assert result["model_campaign"] == prereg["campaign"]
    assert design["uses_cross_map"] is False
    assert design["splits"]["train"]["n_shards"] == prereg["simulation_counts"]["train"]
    assert design["splits"]["val"]["n_shards"] == prereg["simulation_counts"][
        "validation"
    ]
    assert design["splits"]["audit"]["n_shards"] == prereg["simulation_counts"]["audit"]

    map_counts = {}
    for seed in range(7):
        count = len(list((campaign / "arabis_maps" / f"seed{seed}").glob("*.npz")))
        if count != 32:
            raise RuntimeError(f"seed{seed} has {count} maps, expected 32")
        map_counts[f"seed{seed}"] = count
    ensemble_count = len(list((campaign / "arabis_maps" / "ensemble").glob("*.npz")))
    if ensemble_count != 32:
        raise RuntimeError(f"ensemble has {ensemble_count} maps, expected 32")
    map_counts["ensemble"] = ensemble_count

    for member in frozen["members"]:
        checkpoint = Path(member["checkpoint"])
        stats = Path(member["stats"])
        if sha256(checkpoint) != member["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {checkpoint}")
        if sha256(stats) != member["stats_sha256"]:
            raise RuntimeError(f"stats hash mismatch: {stats}")

    payload = {
        "schema_version": 1,
        "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "campaign": prereg["campaign"],
        "complete": True,
        "simulation_gate_passed": True,
        "arabis_cross_map_used_for_model_selection": False,
        "map_counts": map_counts,
        "artifact_sha256": {
            name: sha256(path) for name, path in sorted(required.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
