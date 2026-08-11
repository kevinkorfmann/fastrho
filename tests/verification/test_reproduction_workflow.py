from __future__ import annotations

import json
import runpy

import pytest

from .inventory import ROOT, referenced_graphics

MODULE = runpy.run_path(str(ROOT / "reproduce/paper.py"))
build_inventory = MODULE["build_inventory"]
load_workflow = MODULE["load_workflow"]
selected_stages = MODULE["selected_stages"]


@pytest.fixture(scope="module")
def workflow() -> dict[str, object]:
    return load_workflow()


def test_default_profile_is_complete_and_ordered(workflow: dict[str, object]) -> None:
    expected = [
        "manuscript",
        "preflight",
        "derived",
        "downloads",
        "figures",
        "stage",
        "audit",
        "pdfs",
        "paper-tests",
        "release-gate",
    ]
    assert workflow["profiles"][workflow["default_profile"]] == expected
    assert [stage["id"] for stage in selected_stages(workflow, "paper", None)] == expected
    assert [
        stage["id"] for stage in selected_stages(workflow, "paper", ["pdfs", "figures", "audit"])
    ] == ["figures", "audit", "pdfs"]


def test_declared_commands_outputs_and_betty_launchers_exist(workflow: dict[str, object]) -> None:
    for stage in workflow["stages"]:
        for command in stage.get("commands", []):
            for part in command:
                if isinstance(part, str) and "/" in part and not part.startswith("-"):
                    candidate = ROOT / part
                    if candidate.suffix in {".py", ".sh"}:
                        assert candidate.is_file(), (stage["id"], part)
        for output in stage.get("outputs", []):
            assert isinstance(output, str) and output

    for item in workflow["betty_slurm_workflows"]:
        assert (ROOT / item["submit"]).is_file()
        assert (ROOT / item["guide"]).is_file()
        submit_text = (ROOT / item["submit"]).read_text(encoding="utf-8")
        assert "sbatch" in submit_text


def test_inventory_covers_every_data_and_figure_producer(workflow: dict[str, object]) -> None:
    inventory = build_inventory(workflow)
    inventory_paths = {record["path"] for record in inventory["scripts"]}
    data = json.loads((ROOT / "paper/data_provenance.yaml").read_text(encoding="utf-8"))
    figures = json.loads((ROOT / "paper/figure_provenance.json").read_text(encoding="utf-8"))
    data_producers = {script for dataset in data["datasets"] for script in dataset["producing_scripts"]}
    figure_producers = {figure["producer"] for figure in figures["figures"]}
    assert data_producers <= inventory_paths
    assert figure_producers <= inventory_paths
    assert all(record["exists"] for record in inventory["scripts"])
    assert inventory["summary"]["active_figures"] == 15


def test_figure_manifest_covers_every_manuscript_include() -> None:
    figures = json.loads((ROOT / "paper/figure_provenance.json").read_text(encoding="utf-8"))
    outputs = {figure["manuscript_target"] for figure in figures["figures"]}
    included = {raw for _, raw in referenced_graphics()}
    assert included == outputs
