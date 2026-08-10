"""Exhaustive executable provenance checks for every included figure."""

from __future__ import annotations

import json
import re
import runpy

from .inventory import ROOT, referenced_graphics

MODULE = runpy.run_path(str(ROOT / "scripts" / "build_manuscript_figures.py"))
MANIFEST = MODULE["MANIFEST"]
build_manifest = MODULE["build_manifest"]


def test_committed_figure_manifest_is_fresh() -> None:
    assert json.loads(MANIFEST.read_text(encoding="utf-8")) == build_manifest()


def test_figure_build_uses_a_fixed_source_date_epoch() -> None:
    manifest = build_manifest()
    epoch = manifest["deterministic_build_environment"]["SOURCE_DATE_EPOCH"]
    assert epoch.isdigit()
    assert int(epoch) > 0


def test_every_included_graphic_has_exactly_one_executable_producer() -> None:
    registered = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    outputs = [row["manuscript_target"] for row in registered]
    included = {target for _, target in referenced_graphics()}
    assert set(outputs) == included
    assert len(outputs) == len(set(outputs))
    for row in registered:
        assert (ROOT / row["producer"]).is_file()
        assert row["command"] or row.get("producer", "").endswith(".svg")
        assert row["inputs"]


def test_figure_manifest_contains_no_manuscript_output_as_an_input() -> None:
    registered = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    outputs = {row["output"] for row in registered}
    for row in registered:
        assert not outputs.intersection(record["path"] for record in row["inputs"])


def test_active_figure_producers_are_checkout_portable() -> None:
    registered = json.loads(MANIFEST.read_text(encoding="utf-8"))["figures"]
    producers = {row["producer"] for row in registered}
    for producer in producers:
        source = (ROOT / producer).read_text(encoding="utf-8")
        assert re.search(r"/(?:Users|home)/[^\s'\"]+", source) is None, (
            f"active figure producer contains a personal absolute path: {producer}"
        )
    for row in registered:
        assert not any(str(part).startswith("/") for part in row["command"])
