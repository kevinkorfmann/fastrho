"""Release-contract tests for the plot-ready paper data downloads."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "docs" / "data" / "downloads"


def test_paper_data_exports_are_current_and_deterministic(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_paper_data.py"), "--output", str(tmp_path)],
        cwd=ROOT,
        check=True,
    )
    expected = {path.name for path in DOWNLOADS.iterdir() if path.is_file()}
    observed = {path.name for path in tmp_path.iterdir() if path.is_file()}
    assert observed == expected
    for name in expected:
        assert (tmp_path / name).read_bytes() == (DOWNLOADS / name).read_bytes(), name


def test_manifest_matches_downloads() -> None:
    manifest = json.loads((DOWNLOADS / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    expected_dataset_count = 11 + int(
        (ROOT / "paper" / "figdata" / "demography_matched_predictions.npz").is_file()
    )
    assert len(manifest["datasets"]) == expected_dataset_count
    for dataset in manifest["datasets"]:
        path = DOWNLOADS / dataset["file"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == dataset["sha256"]
        assert len(payload) == dataset["bytes"]
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            assert next(reader) == dataset["columns"]
            assert sum(1 for _ in reader) == dataset["rows"]
    for resource in manifest["resources"]:
        path = DOWNLOADS / resource["file"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == resource["sha256"]
        assert len(payload) == resource["bytes"]


def test_documentation_links_every_download() -> None:
    page = (ROOT / "docs" / "data.md").read_text()
    manifest = json.loads((DOWNLOADS / "manifest.json").read_text())
    for dataset in manifest["datasets"]:
        assert dataset["file"] in page
    for resource in manifest["resources"]:
        assert resource["file"] in page
    assert "fastrho_paper_data.zip" in page
    assert "manifest.json" in page
    assert "data" in (ROOT / "docs" / "index.md").read_text()


def test_inferred_map_downloads_are_easy_to_find_and_scaled() -> None:
    page = (ROOT / "docs" / "data.md").read_text()
    map_files = {
        "anopheles_maps.tsv.gz",
        "arabis_cross_maps.tsv.gz",
        "arabidopsis_maps.tsv.gz",
        "redpoll_maps.tsv.gz",
        "tree_of_life_maps.tsv.gz",
        "canid_example_map.tsv.gz",
    }
    assert "## Inferred-map downloads" in page
    for filename in map_files:
        assert filename in page
    for scale in ("rho_per_bp", "rate_per_bp", "cM_per_Mb", "_relative_rate"):
        assert scale in page
    assert "mean 1" in page
    assert "10^8" in page
    readme = (ROOT / "README.md").read_text()
    assert "kevinkorfmann.github.io/fastrho/data.html#inferred-map-downloads" in readme


def test_documentation_workflow_publishes_the_built_site() -> None:
    workflow = (ROOT / ".github" / "workflows" / "docs.yml").read_text()
    assert "sphinx-build -W --keep-going -b html" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "pages: write" in workflow
    assert "scripts/smoke_fresh_examples.py" in workflow
    assert 'pip install ".[io]"' in workflow
    assert "needs: [examples, build]" in workflow


def test_complete_bundle_is_readable_and_contains_every_declared_artifact() -> None:
    manifest = json.loads((DOWNLOADS / "manifest.json").read_text())
    result_files = {
        f"results/{path.relative_to(ROOT / 'paper' / 'results_snapshot')}"
        for path in (ROOT / "paper" / "results_snapshot").rglob("*.json")
    }
    expected = {
        "README.txt",
        "manifest.json",
        *(f"tables/{row['file']}" for row in manifest["datasets"]),
        *(f"resources/{row['file']}" for row in manifest["resources"]),
        *result_files,
    }
    with zipfile.ZipFile(DOWNLOADS / "fastrho_paper_data.zip") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == expected
        assert archive.read("manifest.json") == (DOWNLOADS / "manifest.json").read_bytes()
        for row in manifest["datasets"]:
            assert archive.read(f"tables/{row['file']}") == (DOWNLOADS / row["file"]).read_bytes()
        for row in manifest["resources"]:
            assert archive.read(f"resources/{row['file']}") == (DOWNLOADS / row["file"]).read_bytes()
        readme = archive.read("README.txt").decode("utf-8")
    for scale in ("rho_per_bp", "rate_per_bp", "cM_per_Mb", "_relative_rate"):
        assert scale in readme
    assert "mean 1" in readme


def test_active_phase2_release_has_expected_public_scope() -> None:
    manifest = json.loads((DOWNLOADS / "manifest.json").read_text())
    rows = {row["file"]: row["rows"] for row in manifest["datasets"]}
    assert rows["anopheles_maps.tsv.gz"] == 41_463
    assert rows["phase2_2la.tsv.gz"] == 9
    assert rows["phase2_resistance.tsv.gz"] == 135
    assert rows["phase2_pedigree_windows.tsv.gz"] == 43
    assert rows["phase2_pyrho.tsv.gz"] == 3
    pedigree = next(
        row for row in manifest["datasets"] if row["file"] == "phase2_pedigree_windows.tsv.gz"
    )
    assert "inferred_normalized" in pedigree["columns"]
    assert "atlas_normalized" not in pedigree["columns"]
    with zipfile.ZipFile(DOWNLOADS / "phase2_results.zip") as archive:
        assert archive.testzip() is None
        assert "results/phase2_resistance.json" in archive.namelist()
        assert "release/atlas_anopheles/manifest.tsv" in archive.namelist()


def test_mosquito_map_scale_is_arm_specific_and_reproducible() -> None:
    with gzip.open(DOWNLOADS / "anopheles_maps.tsv.gz", mode="rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows
    assert "Ne_used" in rows[0]
    scales: dict[tuple[str, str], set[float]] = {}
    for row in rows:
        ne_used = float(row["Ne_used"])
        rate = float(row["rate_per_bp"])
        rho = float(row["rho_per_bp"])
        cm_per_mb = float(row["cM_per_Mb"])
        scales.setdefault((row["cohort"], row["chromosome_arm"]), set()).add(ne_used)
        assert math.isclose(rate, rho / (4 * ne_used), rel_tol=2e-5)
        assert math.isclose(cm_per_mb, rate * 1e8, rel_tol=5e-4, abs_tol=5e-5)
    assert all(len(values) == 1 for values in scales.values())
    assert len(scales) == 45
    assert len({next(iter(values)) for values in scales.values()}) > 9


def test_mosquito_download_is_self_documenting() -> None:
    with zipfile.ZipFile(DOWNLOADS / "anopheles_maps.zip") as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {"README.txt", "anopheles_maps.tsv.gz"}
        assert archive.read("anopheles_maps.tsv.gz") == (
            DOWNLOADS / "anopheles_maps.tsv.gz"
        ).read_bytes()
        readme = archive.read("README.txt").decode("utf-8")
    for required in ("0-based", "AgamP4", "rho_per_bp / (4 * Ne_used)", "Do not divide", "Ne_target"):
        assert required in readme


def test_public_bundles_exclude_restricted_anopheles_material() -> None:
    forbidden_names = ("ag3", "agam", "arabiensis")
    forbidden_text = (b"Ag3", b"Ag1000G phase 3", b"Aarabiensis")
    for filename in ("fastrho_paper_data.zip", "phase2_results.zip"):
        with zipfile.ZipFile(DOWNLOADS / filename) as archive:
            for name in archive.namelist():
                assert not any(marker in name.lower() for marker in forbidden_names), name
                if Path(name).suffix.lower() in {".json", ".txt", ".tsv", ".md"}:
                    payload = archive.read(name)
                    assert not any(marker in payload for marker in forbidden_text), name


def test_nested_demography_input_bundle_is_readable() -> None:
    with zipfile.ZipFile(DOWNLOADS / "demography_matched_inputs.zip") as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert names
        assert "bottleneck_n20/config.json" in names
        assert "expansion_n20/config.json" in names
        assert any(name.endswith(".trees") for name in names)
        assert any(name.endswith(".vcf") for name in names)
        assert any(name.endswith(".npz") for name in names)
