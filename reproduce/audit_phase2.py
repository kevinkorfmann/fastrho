#!/usr/bin/env python3
"""Audit canonical Phase 2 includes, producers, hashes, and numeric inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_MANUSCRIPT = ROOT / "tmp" / "reproduce" / "manuscript"
OUTPUT = HERE / "audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def includes(path: Path, command: str) -> set[str]:
    pattern = re.compile(rf"\\{command}(?:\[[^]]*\])?\{{([^}}]+)\}}")
    return set(pattern.findall(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--check-published", action="store_true")
    args = parser.parse_args()
    manuscript = args.manuscript.expanduser().resolve()
    lock = json.loads((HERE / "manuscript.lock.json").read_text(encoding="utf-8"))
    registry = json.loads((HERE / "artifacts.json").read_text(encoding="utf-8"))
    source_registry = json.loads((HERE / "source_bundles.json").read_text(encoding="utf-8"))
    figures = includes(manuscript / "main_phase2.tex", "includegraphics") | includes(
        manuscript / "si_phase2.tex", "includegraphics"
    )
    tex_inputs = includes(manuscript / "main_phase2.tex", "input") | includes(
        manuscript / "si_phase2.tex", "input"
    )
    declared = {record["manuscript_target"] for record in registry["artifacts"]}
    expected = figures | {
        target if Path(target).suffix else f"{target}.tex" for target in tex_inputs
    }
    expected.update(
        record["manuscript_target"]
        for record in registry["artifacts"]
        if record["role"] == "si-bibliography"
    )
    errors = []
    if expected - declared:
        errors.append(f"undeclared manuscript artifacts: {sorted(expected - declared)}")
    if declared - expected:
        errors.append(f"declared artifacts not included by Phase 2: {sorted(declared - expected)}")

    records = []
    for artifact in registry["artifacts"]:
        source = ROOT / artifact["package_output"]
        producer = ROOT / artifact["producer"]
        target = manuscript / artifact["manuscript_target"]
        expected_hash = lock["files"].get(artifact["manuscript_target"])
        source_hash = sha256(source) if source.is_file() else None
        target_hash = sha256(target) if target.is_file() else None
        published_match = source_hash == expected_hash
        if not source.is_file():
            errors.append(f"missing package output: {artifact['package_output']}")
        if not producer.is_file():
            errors.append(f"missing producer: {artifact['producer']}")
        if not target.is_file():
            errors.append(f"missing staged target: {artifact['manuscript_target']}")
        if args.check_published and not published_match:
            errors.append(f"package output differs from locked publication: {artifact['package_output']}")
        records.append(
            {
                **artifact,
                "published_sha256": expected_hash,
                "package_sha256": source_hash,
                "staged_sha256": target_hash,
                "matches_published": published_match,
            }
        )

    os.environ["FASTRHO_MANUSCRIPT_ROOT"] = str(manuscript)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.verification.inventory import reader_numeric_tokens  # noqa: PLC0415

    tokens = reader_numeric_tokens()
    bundle_files: dict[str, list[dict[str, str]]] = {}
    for bundle, patterns in source_registry["bundles"].items():
        found: set[Path] = set()
        for pattern in patterns:
            matches = {path for path in ROOT.glob(pattern) if path.is_file()}
            if not matches:
                errors.append(f"source-bundle pattern matched nothing: {bundle}: {pattern}")
            found.update(matches)
        bundle_files[bundle] = [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in sorted(found)
        ]
    artifact_by_target = {
        record["manuscript_target"]: record for record in registry["artifacts"]
    }
    numbers = []
    for token in tokens:
        binding: dict[str, object] | None = None
        if token.file.startswith("manuscript/"):
            target = token.file.removeprefix("manuscript/")
            artifact = artifact_by_target.get(target)
            if artifact:
                binding = {
                    "kind": "generated-artifact",
                    "artifact": target,
                    "producer": artifact["producer"],
                }
        else:
            for rule in source_registry["rules"]:
                if re.fullmatch(rule["file"], token.file) and re.search(
                    rule["section_regex"], token.section, flags=re.IGNORECASE
                ):
                    binding = {"kind": "source-bundle", "bundle": rule["bundle"]}
                    break
        if binding is None:
            errors.append(f"unbound number: {token.file}:{token.line}:{token.raw}")
        numbers.append(
            {
                "file": token.file,
                "line": token.line,
                "column": token.column,
                "section": token.section,
                "printed": token.raw,
                "unit": token.unit,
                "context": token.context,
                "binding": binding,
            }
        )
    audit = {
        "schema_version": 1,
        "manuscript_repository": lock["repository"],
        "manuscript_commit": lock["commit"],
        "scope": "Phase 2 only",
        "summary": {
            "figures": len(figures),
            "generated_tex_inputs": len(tex_inputs),
            "declared_artifacts": len(records),
            "numeric_occurrences_inventoried": len(tokens),
            "numeric_occurrences_bound": sum(row["binding"] is not None for row in numbers),
            "artifacts_matching_published_bytes": sum(row["matches_published"] for row in records),
        },
        "numeric_policy": source_registry["policy"],
        "source_bundles": bundle_files,
        "numbers": numbers,
        "artifacts": records,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise SystemExit("Phase 2 audit failed:\n- " + "\n- ".join(errors))
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
