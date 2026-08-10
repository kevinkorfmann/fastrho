#!/usr/bin/env python3
"""Audit the active Phase 2 Anopheles analysis and promoted artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ROOT / "paper" / "anopheles_variants"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_phase2(require_complete: bool) -> list[str]:
    errors = []
    root = VARIANTS / "phase2"
    config = json.loads((root / "config.json").read_text())
    selection = root / "cohorts" / "selection.tsv"
    selected = root / "cohorts" / "selected_samples.tsv"
    for path in (selection, selected, root / "provenance" / "source_manifest.json"):
        if not path.is_file():
            errors.append(f"missing Phase 2 design artifact: {path}")
    if selection.is_file() and selected.is_file():
        with selection.open(newline="", encoding="utf-8") as handle:
            cohort_rows = list(csv.DictReader(handle, delimiter="\t"))
        with selected.open(newline="", encoding="utf-8") as handle:
            sample_rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(cohort_rows) != 9:
            errors.append(f"expected 9 Phase 2 primary cohorts, found {len(cohort_rows)}")
        for row in cohort_rows:
            count = sum(item["cohort"] == row["cohort"] for item in sample_rows)
            if count != 40 or int(row["n_diploid"]) != 40:
                errors.append(f"{row['cohort']} does not contain exactly 40 frozen diploids")
        availability = root / "provenance" / "remote" / "five_arm_samples.txt"
        if availability.is_file():
            eligible = set(availability.read_text().splitlines())
            missing = sorted({row["sample_id"] for row in sample_rows} - eligible)
            if missing:
                errors.append(f"selected samples absent from the five-arm intersection: {missing[:5]}")
        else:
            errors.append(f"missing Phase 2 arm-availability artifact: {availability}")
    if require_complete:
        if config["status"] != "complete" or not config["submission_eligible"]:
            errors.append("Phase 2 config has not passed the submission gate")
        required_dirs = ("maps", "results", "figdata", "figures", "tables", "manuscript", "release")
        for name in required_dirs:
            path = root / name
            if not path.is_dir() or not any(item.is_file() for item in path.rglob("*")):
                errors.append(f"incomplete Phase 2 artifact directory: {path}")
        maps = list((root / "maps").glob("*.npz"))
        if len(maps) != 45:
            errors.append(f"expected 45 Phase 2 map files, found {len(maps)}")
        required_results = (
            "results/phase2_map_qc.json",
            "results/phase2_2la.json",
            "results/phase2_resistance.json",
            "results/phase2_pyrho.json",
            "results/pedigree/phase2_pedigree.json",
            "release/atlas_anopheles/manifest.tsv",
        )
        for relative in required_results:
            if not (root / relative).is_file():
                errors.append(f"missing promoted Phase 2 artifact: {root / relative}")
        promoted_manifest = root / "provenance" / "promoted_files.sha256"
        if not promoted_manifest.is_file():
            errors.append(f"missing promoted-artifact checksum manifest: {promoted_manifest}")
        else:
            for line_number, line in enumerate(promoted_manifest.read_text().splitlines(), 1):
                try:
                    expected, relative = line.split("  ", 1)
                except ValueError:
                    errors.append(f"malformed checksum line {line_number}: {promoted_manifest}")
                    continue
                path = root / relative
                if not path.is_file():
                    errors.append(f"missing checksum-bound Phase 2 artifact: {path}")
                elif sha256(path) != expected:
                    errors.append(f"checksum mismatch in promoted Phase 2 artifact: {path}")
        source_manifest = root / "provenance" / "source_code.sha256"
        if not source_manifest.is_file():
            errors.append(f"missing Phase 2 source-code checksum manifest: {source_manifest}")
        else:
            for line_number, line in enumerate(source_manifest.read_text().splitlines(), 1):
                try:
                    expected, relative = line.split("  ", 1)
                except ValueError:
                    errors.append(f"malformed checksum line {line_number}: {source_manifest}")
                    continue
                path = ROOT / relative
                if not path.is_file():
                    errors.append(f"missing checksum-bound Phase 2 source: {path}")
                elif sha256(path) != expected:
                    errors.append(f"checksum mismatch in Phase 2 source: {path}")
        forbidden = ("Ag3", "Aarabiensis", "13 populations", "15 crosses", "five held-out")
        for path in (root / "fragments").glob("*.tex"):
            text = path.read_text()
            for phrase in forbidden:
                if phrase in text:
                    errors.append(f"restricted-release phrase {phrase!r} remains in {path}")
        for path in (root / "fragments").glob("*.tex"):
            if "PENDING" in path.read_text() or "TODO" in path.read_text():
                errors.append(f"unresolved manuscript placeholder in {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-phase2-complete", action="store_true")
    args = parser.parse_args()
    errors = audit_phase2(args.require_phase2_complete)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Phase 2 design: verified" if not args.require_phase2_complete else "Phase 2 submission: verified")


if __name__ == "__main__":
    main()
