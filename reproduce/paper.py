#!/usr/bin/env python3
"""One ordered entry point for reproducing the fastrho manuscript and SI."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = Path(__file__).with_name("workflow.json")
REQUIRED_EXECUTABLES = ("bash", "latexmk", "uv")
STAGED_MANUSCRIPT = ROOT / "tmp" / "reproduce" / "manuscript"


def load_workflow() -> dict[str, object]:
    """Load the machine-readable workflow and reject a malformed top-level shape."""

    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    required = {"schema_version", "profiles", "stages", "betty_slurm_workflows"}
    missing = required - workflow.keys()
    if missing:
        raise ValueError(f"workflow is missing required fields: {sorted(missing)}")
    return workflow


def _stage_index(workflow: dict[str, object]) -> dict[str, dict[str, object]]:
    stages = workflow["stages"]
    if not isinstance(stages, list):
        raise TypeError("workflow stages must be a list")
    indexed: dict[str, dict[str, object]] = {}
    for stage in stages:
        if not isinstance(stage, dict) or not isinstance(stage.get("id"), str):
            raise TypeError("every workflow stage must have a string id")
        if stage["id"] in indexed:
            raise ValueError(f"duplicate workflow stage: {stage['id']}")
        indexed[stage["id"]] = stage
    return indexed


def selected_stages(
    workflow: dict[str, object], profile: str, only: list[str] | None
) -> list[dict[str, object]]:
    """Return validated stages in their declared execution order."""

    indexed = _stage_index(workflow)
    profiles = workflow["profiles"]
    if not isinstance(profiles, dict) or profile not in profiles:
        choices = ", ".join(sorted(profiles)) if isinstance(profiles, dict) else ""
        raise ValueError(f"unknown profile {profile!r}; choose one of: {choices}")
    if only:
        requested = set(only)
        stage_ids = [stage_id for stage_id in indexed if stage_id in requested]
        unknown = [stage_id for stage_id in only if stage_id not in indexed]
    else:
        stage_ids = profiles[profile]
        unknown = [stage_id for stage_id in stage_ids if stage_id not in indexed]
    if unknown:
        raise ValueError(f"unknown workflow stages: {', '.join(unknown)}")
    return [indexed[stage_id] for stage_id in stage_ids]


def build_inventory(workflow: dict[str, object]) -> dict[str, object]:
    """Assemble the live analysis-script inventory from the canonical ledgers."""

    data = json.loads((ROOT / "paper" / "data_provenance.yaml").read_text(encoding="utf-8"))
    figures = json.loads((ROOT / "paper" / "figure_provenance.json").read_text(encoding="utf-8"))
    datasets_by_script: dict[str, set[str]] = {}
    for dataset in data["datasets"]:
        for script in dataset["producing_scripts"]:
            datasets_by_script.setdefault(script, set()).add(dataset["id"])

    figures_by_script: dict[str, set[str]] = {}
    for figure in figures["figures"]:
        producer = figure["producer"]
        figures_by_script.setdefault(producer, set()).add(figure["output"])

    portable_scripts: set[str] = {"reproduce/paper.py", "reproduce/run.sh"}
    for stage in workflow["stages"]:
        for command in stage.get("commands", []):
            portable_scripts.update(
                part for part in command if isinstance(part, str) and (ROOT / part).is_file()
            )

    all_scripts = sorted(set(datasets_by_script) | set(figures_by_script) | portable_scripts)
    records = []
    for script in all_scripts:
        records.append(
            {
                "path": script,
                "exists": (ROOT / script).is_file(),
                "datasets": sorted(datasets_by_script.get(script, set())),
                "figures": sorted(figures_by_script.get(script, set())),
                "portable_workflow": script in portable_scripts,
            }
        )
    return {
        "schema_version": 1,
        "source_ledgers": ["paper/data_provenance.yaml", "paper/figure_provenance.json"],
        "summary": {
            "scripts": len(records),
            "dataset_producers": len(datasets_by_script),
            "figure_producers": len(figures_by_script),
            "active_figures": len(figures["figures"]),
            "datasets": len(data["datasets"]),
        },
        "scripts": records,
        "betty_slurm_workflows": workflow["betty_slurm_workflows"],
    }


def preflight(workflow: dict[str, object], *, check_tools: bool = True) -> None:
    """Fail early if a declared executable, producer, launcher, guide, or ledger is absent."""

    errors: list[str] = []
    if check_tools:
        for executable in REQUIRED_EXECUTABLES:
            if shutil.which(executable) is None:
                errors.append(f"required executable is not on PATH: {executable}")

    _stage_index(workflow)
    for profile, stage_ids in workflow["profiles"].items():
        try:
            selected_stages(workflow, profile, stage_ids)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))

    for ledger in workflow.get("canonical_ledgers", []):
        if not (ROOT / ledger).is_file():
            errors.append(f"canonical ledger is missing: {ledger}")

    for item in workflow["betty_slurm_workflows"]:
        for field in ("submit", "guide"):
            path = item.get(field)
            if not isinstance(path, str) or not (ROOT / path).is_file():
                errors.append(f"Betty workflow {item.get('id')} lacks {field}: {path}")

    inventory = build_inventory(workflow)
    for record in inventory["scripts"]:
        if not record["exists"]:
            errors.append(f"active producer is missing: {record['path']}")

    if errors:
        raise RuntimeError("paper reproduction preflight failed:\n- " + "\n- ".join(errors))
    summary = inventory["summary"]
    print(
        "preflight passed: "
        f"{summary['scripts']} active scripts, {summary['datasets']} datasets, "
        f"{summary['active_figures']} figures"
    )


def _run_command(command: list[str], environment: dict[str, str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def run_stages(stages: list[dict[str, object]], workflow: dict[str, object], dry_run: bool) -> None:
    """Run selected stages sequentially and verify every declared output."""

    environment = dict(os.environ)
    environment.setdefault("SOURCE_DATE_EPOCH", "1420070400")
    environment.setdefault("FASTRHO_MANUSCRIPT_ROOT", str(STAGED_MANUSCRIPT))
    started = time.monotonic()
    for number, stage in enumerate(stages, 1):
        print(f"\n[{number}/{len(stages)}] {stage['id']}: {stage['description']}", flush=True)
        if stage.get("internal") == "preflight":
            if dry_run:
                print("  internal: preflight")
            else:
                preflight(workflow)
        for command in stage.get("commands", []):
            if dry_run:
                print(f"  $ {shlex.join(command)}")
            else:
                _run_command(command, environment)
        if not dry_run:
            missing = [output for output in stage.get("outputs", []) if not (ROOT / output).is_file()]
            if missing:
                raise RuntimeError(
                    f"stage {stage['id']} did not create declared outputs: {', '.join(missing)}"
                )
    if not dry_run:
        elapsed = time.monotonic() - started
        print(f"\npaper reproduction completed in {elapsed:.1f} seconds")


def _parse_only(value: str | None) -> list[str] | None:
    if value is None:
        return None
    stages = [item.strip() for item in value.split(",") if item.strip()]
    return stages or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="print the ordered commands without running")
    plan_parser.add_argument("--profile", default="paper")
    plan_parser.add_argument("--only", help="comma-separated stage ids")

    run_parser = subparsers.add_parser("run", help="run an ordered reproduction profile")
    run_parser.add_argument("--profile", default="paper")
    run_parser.add_argument("--only", help="comma-separated stage ids")

    inventory_parser = subparsers.add_parser(
        "inventory", help="print every active data and figure producer as JSON"
    )
    inventory_parser.add_argument("--compact", action="store_true")

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate the workflow without rebuilding artifacts"
    )
    preflight_parser.add_argument(
        "--metadata-only", action="store_true", help="skip local executable discovery"
    )
    args = parser.parse_args(argv)
    workflow = load_workflow()

    try:
        if args.command == "inventory":
            indent = None if args.compact else 2
            print(json.dumps(build_inventory(workflow), indent=indent, sort_keys=True))
        elif args.command == "preflight":
            preflight(workflow, check_tools=not args.metadata_only)
        else:
            stages = selected_stages(workflow, args.profile, _parse_only(args.only))
            run_stages(stages, workflow, dry_run=args.command == "plan")
    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
