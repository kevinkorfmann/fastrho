"""Keep every model named in the Phase 2 SI connected to public training code."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

WORKFLOWS = {
    "base-v1": ("scripts/train_base_view.sh", "scripts/train_base_view.sh"),
    "domain-randomized-v1": (
        "models/domain-randomized-v1/reproduce/submit.sh",
        "models/domain-randomized-v1/reproduce/slurm/00_simulate.sbatch",
    ),
    "composite-ld-v1": ("scripts/train_base_view.sh", "scripts/train_base_view.sh"),
    "high-ne-v1": ("scripts/retrain_highne.sh", "fastrho/simulate.py"),
    "selfing-v1": ("scripts/selfing_train.sh", "scripts/selfing_gen.py"),
    "dog-bottleneck-v1": ("scripts/dog_train_bottleneck.sh", "scripts/dog_gen.py"),
    "arabis-smalln-ensemble": (
        "research/arabis/slurm_smalln/submit.sh",
        "scripts/arabis_smalln_selfing_gen.py",
    ),
    "arabis-structured-ensemble": (
        "research/arabis/slurm_structured/submit.sh",
        "scripts/arabis_structured_selfing_gen.py",
    ),
    "canid-structure-paper-analysis": (
        "scripts/wolf_structure_train.sh",
        "scripts/wolf_structure_gen.py",
    ),
}


def test_every_phase2_model_has_training_and_simulation_code() -> None:
    missing = [
        f"{model}: {path}"
        for model, paths in WORKFLOWS.items()
        for path in paths
        if not (ROOT / path).is_file()
    ]
    assert not missing, "missing public workflow files:\n" + "\n".join(missing)


def test_released_model_manifests_match_paper_region_counts() -> None:
    expected = {
        "base-v1": 15000,
        "domain-randomized-v1": 15000,
        "composite-ld-v1": 15000,
        "high-ne-v1": 15000,
        "selfing-v1": 4000,
        "dog-bottleneck-v1": 15000,
    }
    observed = {}
    for model in expected:
        manifest = json.loads((ROOT / "models" / model / "manifest.json").read_text())
        observed[model] = manifest["training"]["regions"]
    assert observed == expected


def test_portable_specialist_entrypoints_do_not_name_original_hosts() -> None:
    paths = {
        path
        for model, pair in WORKFLOWS.items()
        if model not in {"domain-randomized-v1", "arabis-smalln-ensemble", "arabis-structured-ensemble"}
        for path in pair[:1]
    }
    forbidden = ("/home/kkor", "ssh betty", "ssh sesame")
    failures = []
    for path in sorted(paths):
        text = (ROOT / path).read_text()
        for token in forbidden:
            if token in text:
                failures.append(f"{path}: {token}")
    assert not failures, "host-bound training entry points:\n" + "\n".join(failures)
