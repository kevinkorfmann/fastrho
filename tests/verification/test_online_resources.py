"""Opt-in live verification of data sources and DOI-backed citations.

Run with ``pytest tests/verification --run-online -m online``. These checks are
separated from deterministic unit tests because publishers and repositories can
rate-limit or temporarily block automated clients.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import pytest

from .inventory import BIB_BY_KEY, CITED_KEYS, manuscript_urls, provenance_urls

pytestmark = pytest.mark.online

CONTACT_EMAIL = "korfmann@sas.upenn.edu"
USER_AGENT = f"fastrho-manuscript-verifier/1.1 (mailto:{CONTACT_EMAIL})"
LIVE_URLS = tuple(
    sorted(
        url
        for url in set(provenance_urls()) | set(manuscript_urls())
        if urlparse(url).scheme in {"http", "https"}
    )
)
DOI_ENTRIES = tuple(
    BIB_BY_KEY[key] for key in CITED_KEYS if BIB_BY_KEY[key].fields.get("doi")
)
NON_DOI_ENTRIES = tuple(
    BIB_BY_KEY[key] for key in CITED_KEYS if not BIB_BY_KEY[key].fields.get("doi")
)


def _request(url: str, *, accept: str = "*/*") -> tuple[int, bytes]:
    request = Request(
        url,
        headers={
            "Accept": accept,
            "Range": "bytes=0-8191",
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urlopen(request, timeout=20) as response:
            return int(response.status), response.read(2_000_000)
    except HTTPError as error:
        return int(error.code), error.read(2_000_000)
    except URLError as error:
        pytest.fail(f"network error for {url}: {error.reason}")


def _normalize_title(value: str) -> str:
    value = re.sub(r"\\[A-Za-z]+|[{}]", "", value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


@lru_cache(maxsize=None)
def _crossref_record(doi: str) -> tuple[int, bytes]:
    url = (
        f"https://api.crossref.org/works/{quote(doi, safe='')}"
        f"?mailto={quote(CONTACT_EMAIL, safe='@')}"
    )
    for attempt in range(5):
        status, payload = _request(url, accept="application/json")
        if status != 429:
            time.sleep(0.12)
            return status, payload
        time.sleep(2**attempt)
    return status, payload


@pytest.mark.parametrize("url", LIVE_URLS, ids=LIVE_URLS)
def test_each_external_dataset_or_manuscript_url_is_live(url: str) -> None:
    status, _ = _request(url)
    assert 200 <= status < 400, f"external resource returned HTTP {status}: {url}"


@pytest.mark.parametrize("entry", NON_DOI_ENTRIES, ids=[entry.key for entry in NON_DOI_ENTRIES])
def test_each_non_doi_citation_has_a_live_official_url(entry) -> None:
    url = entry.fields["url"].strip()
    status, _ = _request(url)
    assert 200 <= status < 400, f"citation URL returned HTTP {status}: {entry.key} ({url})"


@pytest.mark.parametrize("entry", DOI_ENTRIES, ids=[entry.key for entry in DOI_ENTRIES])
def test_each_doi_citation_matches_crossref_title_and_year(entry) -> None:
    doi = entry.fields["doi"].strip()
    status, payload = _crossref_record(doi)
    assert status == 200, f"Crossref returned HTTP {status} for {entry.key} ({doi})"
    message = json.loads(payload)["message"]

    remote_titles = message.get("title") or []
    assert remote_titles, f"Crossref record for {entry.key} lacks a title"
    local_title = _normalize_title(entry.fields["title"])
    remote_title = _normalize_title(remote_titles[0])
    similarity = SequenceMatcher(None, local_title, remote_title).ratio()
    assert similarity >= 0.90, (
        f"title mismatch for {entry.key}: {local_title!r} != {remote_title!r} ({similarity:.2f})"
    )

    date_parts = []
    for field in ("published-print", "published-online", "published", "issued", "created"):
        record = message.get(field) or {}
        candidate = record.get("date-parts") or []
        if candidate and candidate[0]:
            date_parts = candidate[0]
            break
    assert date_parts, f"Crossref record for {entry.key} lacks a publication date"
    assert abs(int(entry.fields["year"]) - int(date_parts[0])) <= 1, (
        f"year mismatch for {entry.key}: BibTeX={entry.fields['year']}, Crossref={date_parts[0]}"
    )
