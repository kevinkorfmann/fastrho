"""One-record-at-a-time checks for every bibliography entry used by the project."""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from .inventory import (
    BIB_BY_KEY,
    BIB_ENTRIES,
    CITED_KEYS,
    MANUSCRIPT,
    ROOT,
    BibEntry,
    provenance_citations,
    valid_http_url,
)

ENTRY_IDS = [entry.key for entry in BIB_ENTRIES]
ACTIVE_KEYS = set(CITED_KEYS) | {key for _, key in provenance_citations()}
ACTIVE_ENTRIES = tuple(entry for entry in BIB_ENTRIES if entry.key in ACTIVE_KEYS)


@pytest.mark.parametrize("entry", BIB_ENTRIES, ids=ENTRY_IDS)
def test_reference_has_core_metadata(entry: BibEntry) -> None:
    """Every reference is complete, plausible, and syntactically resolvable."""

    required = {"title", "author", "year"}
    missing = sorted(required - entry.fields.keys())
    assert not missing, f"{entry.key} lacks required fields: {missing}"
    for field in required:
        assert entry.fields[field].strip(), f"{entry.key}.{field} is empty"
    year = entry.fields.get("year", "")
    assert re.fullmatch(r"\d{4}", year), f"{entry.key} has noncanonical year {year!r}"
    assert 1800 <= int(year) <= 2026, f"{entry.key} has implausible year {year}"
    flattened = " ".join(entry.fields.values()).lower()
    forbidden = (
        "bibliographic details to be verified",
        "citation needed",
        "fixme",
        "placeholder",
        "tbd",
        "unknown author",
    )
    hits = [token for token in forbidden if token in flattened]
    assert not hits, f"{entry.key} contains unresolved metadata: {hits}"
    if entry.kind == "article":
        assert entry.fields.get("journal", "").strip(), f"{entry.key} lacks journal"
        assert entry.fields.get("pages", "").strip() or entry.fields.get("doi", "").strip(), (
            f"{entry.key} lacks pages/article number and DOI"
        )
    elif entry.kind in {"inproceedings", "conference"}:
        assert entry.fields.get("booktitle", "").strip(), f"{entry.key} lacks booktitle"
    elif entry.kind in {"misc", "online"}:
        assert any(
            entry.fields.get(field, "").strip() for field in ("url", "doi", "howpublished")
        ), f"{entry.key} lacks a stable publication route"
    else:
        assert entry.kind in {"book", "incollection", "phdthesis", "mastersthesis", "techreport"}
    doi = entry.fields.get("doi", "").strip()
    url = entry.fields.get("url", "").strip()
    if doi:
        assert re.fullmatch(r"10\.\d{4,9}/\S+", doi, flags=re.IGNORECASE), (
            f"{entry.key} has malformed DOI {doi!r}"
        )
        assert not doi.lower().startswith("https://doi.org/")
    if url:
        assert valid_http_url(url), f"{entry.key} has malformed URL {url!r}"


@pytest.mark.parametrize("key", CITED_KEYS)
def test_each_manuscript_citation_has_substantive_bibliography_metadata(key: str) -> None:
    """Each active citation resolves to a titled, authored source with a venue route."""

    assert key in BIB_BY_KEY, f"undefined citation {key}"
    entry = BIB_BY_KEY[key]
    assert len(re.sub(r"[{}\\]", "", entry.fields["title"]).split()) >= 2
    assert len(re.sub(r"[{}\\]", "", entry.fields["author"]).split()) >= 2
    assert re.search(rf"\\cite[A-Za-z*]*\{{[^}}]*\b{re.escape(key)}\b[^}}]*\}}", MANUSCRIPT)


@pytest.mark.parametrize("entry", ACTIVE_ENTRIES, ids=[entry.key for entry in ACTIVE_ENTRIES])
def test_each_active_reference_has_a_stable_verification_route(entry: BibEntry) -> None:
    """Every cited work can be checked through a DOI or an official stable URL."""

    assert entry.fields.get("doi", "").strip() or entry.fields.get("url", "").strip(), (
        f"{entry.key} has neither a DOI nor a stable URL"
    )


def test_bibliography_keys_are_unique() -> None:
    duplicates = sorted(key for key, count in Counter(ENTRY_IDS).items() if count != 1)
    assert not duplicates, f"duplicate BibTeX keys: {duplicates}"


def test_dois_are_unique() -> None:
    dois = [entry.fields["doi"].lower() for entry in ACTIVE_ENTRIES if entry.fields.get("doi")]
    duplicates = sorted(doi for doi, count in Counter(dois).items() if count != 1)
    assert not duplicates, f"duplicate DOI records: {duplicates}"


def test_manuscript_metadata_citation_keys_resolve() -> None:
    """External numeric facts must point to real bibliography records."""

    metadata = json.loads((ROOT / "paper" / "manuscript_metadata.json").read_text())
    cited: set[str] = set()

    def collect(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "citation" and isinstance(child, str):
                    cited.add(child)
                elif key == "citations" and isinstance(child, list):
                    cited.update(item for item in child if isinstance(item, str))
                else:
                    collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(metadata)
    assert not (cited - set(BIB_BY_KEY)), (
        f"manuscript metadata has undefined citation keys: {sorted(cited - set(BIB_BY_KEY))}"
    )


def test_reference_keys_do_not_alias_the_same_normalized_title() -> None:
    normalized = [
        re.sub(r"[^a-z0-9]+", "", entry.fields.get("title", "").lower())
        for entry in ACTIVE_ENTRIES
    ]
    duplicates = sorted(title for title, count in Counter(normalized).items() if title and count != 1)
    assert not duplicates, "duplicate works appear under multiple BibTeX keys"
