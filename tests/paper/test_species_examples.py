"""Contracts for the manuscript-wide source and inference examples."""

from __future__ import annotations

import json
import subprocess
import sys

import paperlib as P

EXAMPLES = P.REPO_ROOT / "examples" / "manuscript_species"
MANIFEST = json.loads((EXAMPLES / "species.json").read_text(encoding="utf-8"))
SPECIES = {row["key"]: row for row in MANIFEST["species"]}


def run_example(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=P.REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_scope_matches_the_manuscript() -> None:
    transect = json.loads((P.REPO_ROOT / "paper" / "figdata" / "transect.json").read_text())
    expected_transect = {row["key"]: row for row in transect["species"]}
    actual_transect = {key: row for key, row in SPECIES.items() if "transect" in row["paper_roles"]}
    assert set(actual_transect) == set(expected_transect)
    for key, expected in expected_transect.items():
        assert actual_transect[key]["scientific_name"] == expected["latin"]
        assert actual_transect[key]["common_name"] == expected["common"]
        assert actual_transect[key]["cohort"]["n_individuals"] == expected["n_dip"]

    excluded = {
        key for key, row in SPECIES.items()
        if any(role.startswith("excluded:") for role in row["paper_roles"])
    }
    assert excluded == {
        "dog", "vervet", "buffalo", "yak", "pig", "greattit", "mallard",
        "chicken", "tilapia", "trout", "oyster", "honeybee", "celegans", "beech",
    }
    assert set(SPECIES) == set(expected_transect) | excluded | {
        "wolf", "anopheles_gambiae", "anopheles_coluzzii", "redpoll",
    }
    assert len(SPECIES) == 28
    assert SPECIES["anopheles_gambiae"]["paper_roles"] == ["primary_anopheles"]
    assert SPECIES["anopheles_coluzzii"]["paper_roles"] == ["primary_anopheles"]


def test_every_preset_has_a_complete_scientific_contract() -> None:
    registry = json.loads((P.REPO_ROOT / "fastrho" / "model_registry.json").read_text())
    models = {row["id"]: row for row in registry["models"]}
    for key, row in SPECIES.items():
        source = row["source"]
        inference = row["inference"]
        assert source["landing_page"].startswith("https://")
        assert source["download_kind"] in {"direct", "api", "manual"}
        assert inference["contig"]
        assert inference["mutation_rate"] > 0
        assert inference["window_size_bp"] > 0
        assert inference["input_mode"] in {"phased", "unphased", "unpolarized"}
        if inference["model_id"] is not None:
            assert inference["model_id"] in models
            assert inference["input_mode"] in models[inference["model_id"]]["supported_inputs"]
        else:
            assert key in {"dog", "wolf"}

        if source["download_kind"] == "direct":
            primary = [file for file in source["files"] if file["role"] == "primary"]
            assert len(primary) == 1
            assert primary[0]["url"].startswith("https://")
        else:
            assert source["instructions"]


def test_user_guide_names_every_preset() -> None:
    guide = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(guide.split())
    assert MANIFEST["manuscript_title"] in normalized
    for key in (row["key"] for row in MANIFEST["species"] if "transect" in row["paper_roles"]):
        assert f"`{key}`" in guide
    assert "Phase 2 AR1" in guide
    assert "not distributed" in guide.lower()


def test_source_utility_commands_are_executable() -> None:
    script = EXAMPLES / "data.py"
    listed = run_example(script, "list")
    assert "anopheles_gambiae" in listed.stdout
    shown = run_example(script, "show", "redpoll")
    assert "ScYwTfa_10461" in shown.stdout
    dry_download = run_example(script, "download", "human", "--dry-run")
    assert "human.chr2.vcf.gz" in dry_download.stdout
    phase2_route = run_example(script, "download", "anopheles_gambiae", "--dry-run")
    assert "download_release.sh" in phase2_route.stdout
    assert "extract_phase2_hdf5.py" in phase2_route.stdout
    prepared = run_example(
        script,
        "prepare",
        "human",
        "--vcf",
        "raw.vcf.gz",
        "--out",
        "prepared.vcf.gz",
        "--dry-run",
    )
    assert "bcftools view" in prepared.stdout
    assert 'GT="mis"' in prepared.stdout


def test_inference_dry_run_is_executable() -> None:
    inferred = run_example(
        EXAMPLES / "infer.py",
        "--species",
        "human",
        "--vcf",
        "cohort.vcf.gz",
        "--checkpoint",
        "model.ckpt",
        "--stats",
        "stats.npz",
        "--out",
        "map.bed",
        "--dry-run",
    )
    assert "1.29e-08" in inferred.stdout
    assert "unpolarized" in inferred.stdout

def test_large_example_outputs_are_ignored() -> None:
    ignore = (P.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "examples/manuscript_species/data/" in ignore
    assert "examples/manuscript_species/maps/" in ignore
    assert not (EXAMPLES / "data").exists()
    assert not (EXAMPLES / "maps").exists()
