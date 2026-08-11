"""Data provenance, result serialization, and executable-source contracts."""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from .inventory import (
    BIB_BY_KEY,
    PROVENANCE,
    ROOT,
    local_provenance_paths,
    producing_python_files,
    producing_shell_files,
    provenance_citations,
    provenance_urls,
    relative_id,
    result_json_files,
    result_npz_files,
    valid_http_url,
)

DATASETS = tuple(PROVENANCE["datasets"])
LOCAL_PATHS = local_provenance_paths()
PROVENANCE_CITATIONS = provenance_citations()
PROVENANCE_URLS = provenance_urls()
JSON_FILES = result_json_files()
NPZ_FILES = result_npz_files()
PYTHON_FILES = producing_python_files()
SHELL_FILES = producing_shell_files()


def _walk_json(value: object, location: str = "$" ) -> None:
    """Reject nonportable JSON values recursively."""

    if isinstance(value, dict):
        assert all(isinstance(key, str) and key for key in value), f"bad key at {location}"
        for key, child in value.items():
            _walk_json(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_json(child, f"{location}[{index}]")
    elif isinstance(value, float):
        assert math.isfinite(value), f"nonfinite number at {location}"
    else:
        assert value is None or isinstance(value, (str, int, bool)), (
            f"non-JSON type {type(value).__name__} at {location}"
        )


@pytest.mark.parametrize("source", DATASETS, ids=[source["id"] for source in DATASETS])
def test_each_dataset_has_complete_provenance_schema(source: dict[str, object]) -> None:
    required = {
        "id",
        "name",
        "version",
        "organisms",
        "repository",
        "accession_or_url",
        "terms_url",
        "citation_keys",
        "local_derivatives",
        "producing_scripts",
        "manuscript_scope",
    }
    optional = {"related_terms_urls", "related_sources"}
    optional_scalar = {"primary_2La_validation_scope", "map_panel_scope"}
    assert required <= set(source), f"{source['id']} is missing required provenance fields"
    assert set(source) <= required | optional | optional_scalar, (
        f"{source['id']} has schema drift"
    )
    assert source["manuscript_scope"] in {"primary", "supporting"}
    for field in required - {"organisms", "citation_keys", "local_derivatives", "producing_scripts"}:
        assert isinstance(source[field], (str, int, float)) and str(source[field]).strip()
    for field in ("organisms", "citation_keys", "local_derivatives", "producing_scripts"):
        assert isinstance(source[field], list) and source[field], f"{source['id']}.{field} is empty"
        assert len(source[field]) == len(set(source[field])), f"duplicates in {source['id']}.{field}"
    for field in optional & set(source):
        assert isinstance(source[field], list) and source[field], f"{source['id']}.{field} is empty"
    for field in optional_scalar & set(source):
        assert isinstance(source[field], str) and source[field].strip()


@pytest.mark.parametrize(
    ("source_id", "field", "relative"),
    LOCAL_PATHS,
    ids=[f"{source_id}:{field}:{relative}" for source_id, field, relative in LOCAL_PATHS],
)
def test_each_declared_local_resource_exists_and_is_nonempty(
    source_id: str, field: str, relative: str
) -> None:
    path = ROOT / relative
    assert not Path(relative).is_absolute(), f"{source_id} discloses absolute path {relative}"
    assert path.exists(), f"{source_id}.{field} missing {relative}"
    if path.is_dir():
        assert any(path.iterdir()), f"{source_id}.{field} points to empty directory {relative}"
    else:
        assert path.stat().st_size > 0, f"{source_id}.{field} points to empty file {relative}"


@pytest.mark.parametrize(
    ("source_id", "key"),
    PROVENANCE_CITATIONS,
    ids=[f"{source_id}:{key}" for source_id, key in PROVENANCE_CITATIONS],
)
def test_each_dataset_citation_resolves_to_complete_bibliography_record(
    source_id: str, key: str
) -> None:
    assert key in BIB_BY_KEY, f"{source_id} cites undefined key {key}"
    entry = BIB_BY_KEY[key]
    assert {"title", "author", "year"} <= entry.fields.keys()


@pytest.mark.parametrize("url", PROVENANCE_URLS, ids=PROVENANCE_URLS)
def test_each_dataset_url_is_syntactically_stable(url: str) -> None:
    assert valid_http_url(url), f"malformed provenance URL {url!r}"
    assert not re.search(r"(?:localhost|127\.0\.0\.1|example\.com)", url)


@pytest.mark.parametrize("path", JSON_FILES, ids=[relative_id(path) for path in JSON_FILES])
def test_each_result_json_is_strict_finite_and_nonempty(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    assert raw.strip(), f"empty result file {relative_id(path)}"

    def reject_constant(token: str) -> None:
        raise ValueError(f"nonstandard JSON constant {token}")

    value = json.loads(raw, parse_constant=reject_constant)
    assert isinstance(value, (dict, list)) and value, f"empty top-level result {relative_id(path)}"
    _walk_json(value)


@pytest.mark.parametrize("path", NPZ_FILES, ids=[relative_id(path) for path in NPZ_FILES])
def test_each_result_npz_is_pickle_free_finite_and_nonempty(path: Path) -> None:
    assert path.stat().st_size > 0
    with np.load(path, allow_pickle=False) as archive:
        assert archive.files, f"empty NPZ archive {relative_id(path)}"
        assert len(archive.files) == len(set(archive.files))
        for key in archive.files:
            array = archive[key]
            assert array.dtype.kind != "O", f"unsafe object array {relative_id(path)}:{key}"
            assert array.size > 0, f"empty array {relative_id(path)}:{key}"
            if array.dtype.kind in "fc":
                assert not np.isinf(array).any(), f"infinite array {relative_id(path)}:{key}"
                assert np.isfinite(array).any(), f"entirely missing array {relative_id(path)}:{key}"


@pytest.mark.parametrize("path", PYTHON_FILES, ids=[relative_id(path) for path in PYTHON_FILES])
def test_each_package_or_producing_python_file_parses(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    assert source.strip(), f"empty Python file {relative_id(path)}"
    tree = ast.parse(source, filename=str(path))
    assert isinstance(tree, ast.Module)


@pytest.mark.parametrize("path", SHELL_FILES, ids=[relative_id(path) for path in SHELL_FILES])
def test_each_producing_shell_script_passes_bash_syntax(path: Path) -> None:
    completed = subprocess.run(
        ["bash", "-n", str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr


def test_provenance_dataset_ids_are_unique_and_machine_readable() -> None:
    ids = [source["id"] for source in DATASETS]
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", value) for value in ids)
    assert not [value for value, count in Counter(ids).items() if count != 1]


def test_provenance_access_date_is_iso_and_not_in_the_future() -> None:
    accessed = date.fromisoformat(PROVENANCE["access_date"])
    assert accessed <= date.today()
    assert PROVENANCE["schema_version"] == 1


def test_provenance_policy_states_the_minimum_reproducibility_contract() -> None:
    policy = PROVENANCE["policy"].lower()
    for concept in ("stable accession", "citation", "terms", "producing script"):
        assert concept in policy


def test_exclusion_ledger_is_explicit_even_when_empty() -> None:
    assert "excluded_until_provenance_complete" in PROVENANCE
    assert isinstance(PROVENANCE["excluded_until_provenance_complete"], list)


def test_analysis_readme_exposes_the_complete_ordered_paper_rebuild() -> None:
    guide = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    ordered_entrypoints = (
        "build_manuscript_derived.py",
        "export_paper_data.py",
        "build_manuscript_figures.py",
        "reproduce/stage_manuscript.py",
        "reproduce/audit_phase2.py",
        "reproduce/build_manuscript.py",
        "uv run python -m pytest tests/paper tests/verification -q",
        "release_check.py --strict-models",
    )
    positions = [guide.index(entrypoint) for entrypoint in ordered_entrypoints]
    assert positions == sorted(positions)
    assert "research/demography_matched/" in guide


def test_paper_ci_runs_the_number_figure_data_and_paired_campaign_contracts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "paper-numbers.yml").read_text(
        encoding="utf-8"
    )
    for contract in (
        "tests/verification",
        "test_demography_matched_benchmark.py",
        "test_paper_data_release.py",
        "ruff check fastrho tests",
        "compileall -q fastrho scripts tests",
    ):
        assert contract in workflow
    assert '      - "refs.bib"' in workflow
    assert '      - "paper/refs.bib"' not in workflow
