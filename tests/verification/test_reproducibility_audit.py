"""Artifact-level contracts for the canonical Phase 2 verification."""

from __future__ import annotations

import json

from .inventory import ROOT, referenced_graphics

AUDIT = json.loads((ROOT / "reproduce/audit.json").read_text(encoding="utf-8"))
LOCK = json.loads((ROOT / "reproduce/manuscript.lock.json").read_text(encoding="utf-8"))


def test_audit_targets_the_locked_canonical_manuscript() -> None:
    assert AUDIT["manuscript_commit"] == LOCK["commit"]
    assert AUDIT["manuscript_repository"] == LOCK["repository"]
    assert AUDIT["summary"]["figures"] == 15
    # Only files actually included by the locked manuscript belong in the artifact ledger.
    # The SI's training-prior table is inline, not an input of tables_phase2/prior_envelope.tex.
    assert AUDIT["summary"]["declared_artifacts"] == 21


def test_every_phase2_figure_has_one_declared_artifact() -> None:
    included = {target for _, target in referenced_graphics()}
    declared = {
        record["manuscript_target"]
        for record in AUDIT["artifacts"]
        if record["role"].endswith("figure") or record["role"] == "decorative-static"
    }
    assert included == declared


def test_every_artifact_has_a_live_output_producer_and_hash() -> None:
    for record in AUDIT["artifacts"]:
        assert (ROOT / record["package_output"]).is_file()
        assert (ROOT / record["producer"]).is_file()
        assert len(record["published_sha256"]) == 64
        assert len(record["package_sha256"]) == 64
        assert len(record["staged_sha256"]) == 64
