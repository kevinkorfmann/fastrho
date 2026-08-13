"""Number-level contracts for the canonical Phase 2 verification."""

from __future__ import annotations

import hashlib
import json

from .inventory import ROOT

AUDIT = json.loads((ROOT / "reproduce/audit.json").read_text(encoding="utf-8"))


def test_every_reader_visible_numeric_occurrence_has_a_source_binding() -> None:
    numbers = AUDIT["numbers"]
    assert numbers
    assert len(numbers) == AUDIT["summary"]["numeric_occurrences_inventoried"]
    assert all(record["binding"] for record in numbers)
    assert AUDIT["summary"]["numeric_occurrences_bound"] == len(numbers)


def test_generated_numbers_bind_to_their_exact_artifact_producer() -> None:
    generated = [
        record for record in AUDIT["numbers"]
        if record["file"].startswith("manuscript/")
    ]
    assert generated
    for record in generated:
        assert record["binding"]["kind"] == "generated-artifact"
        assert (ROOT / record["binding"]["producer"]).is_file()


def test_prose_source_bundles_are_checksum_bound() -> None:
    used = {
        record["binding"]["bundle"]
        for record in AUDIT["numbers"]
        if record["binding"]["kind"] == "source-bundle"
    }
    assert used == set(AUDIT["source_bundles"])
    for bundle in used:
        files = AUDIT["source_bundles"][bundle]
        assert files
        for record in files:
            path = ROOT / record["path"]
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_numeric_audit_is_phase2_only() -> None:
    assert AUDIT["scope"] == "Phase 2 only"
    assert all("phase3" not in record["file"].lower() for record in AUDIT["numbers"])
