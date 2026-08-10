"""Deterministic inventories shared by the cross-domain verification suite.

The helpers in this module deliberately use only the Python standard library so
that manuscript verification never depends on the GPU or inference stack.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
_external_manuscript = os.environ.get("FASTRHO_MANUSCRIPT_ROOT")
MANUSCRIPT_DIR = (
    Path(_external_manuscript).expanduser().resolve()
    if _external_manuscript
    else ROOT / "paper" / "manuscript"
)
MAIN_PATH = MANUSCRIPT_DIR / ("main_phase2.tex" if _external_manuscript else "main.tex")
SI_PATH = MANUSCRIPT_DIR / ("si_phase2.tex" if _external_manuscript else "si.tex")
BIB_PATH = MANUSCRIPT_DIR / "refs.bib" if _external_manuscript else ROOT / "refs.bib"
EXTRA_BIB_PATH = MANUSCRIPT_DIR / (
    "generated_phase2/transect_sources.bib" if _external_manuscript else "generated/transect_sources.bib"
)
PROVENANCE_PATH = ROOT / "paper" / "data_provenance.yaml"

MAIN = MAIN_PATH.read_text(encoding="utf-8")
SI = SI_PATH.read_text(encoding="utf-8")
MANUSCRIPT = MAIN + "\n" + SI
PROVENANCE = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class BibEntry:
    """One parsed BibTeX record."""

    kind: str
    key: str
    fields: dict[str, str]
    raw: str


@dataclass(frozen=True)
class NumericToken:
    """One reader-facing numeric literal in the active manuscript sources."""

    file: str
    line: int
    column: int
    raw: str
    prefix: str
    suffix: str
    context: str
    section: str
    unit: str | None


def _balanced_value(text: str, start: int) -> tuple[str, int]:
    """Return a BibTeX field value and the index after it."""

    opener = text[start]
    if opener == "{":
        depth = 0
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1 : index].strip(), index + 1
        raise ValueError("unterminated braced BibTeX value")
    if opener == '"':
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return text[start + 1 : index].strip(), index + 1
        raise ValueError("unterminated quoted BibTeX value")

    end = start
    while end < len(text) and text[end] not in ",\n\r":
        end += 1
    return text[start:end].strip(), end


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cursor = 0
    while cursor < len(body):
        match = re.search(r"(?i)([a-z][a-z0-9_-]*)\s*=\s*", body[cursor:])
        if match is None:
            break
        name = match.group(1).lower()
        start = cursor + match.end()
        if start >= len(body):
            raise ValueError(f"missing value for BibTeX field {name}")
        value, cursor = _balanced_value(body, start)
        fields[name] = value
    return fields


def parse_bibliography(text: str) -> tuple[BibEntry, ...]:
    """Parse top-level BibTeX entries, retaining nested braces in fields."""

    entries: list[BibEntry] = []
    header = re.compile(r"@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,")
    cursor = 0
    while match := header.search(text, cursor):
        depth = 1
        escaped = False
        index = match.end()
        while index < len(text) and depth:
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"unterminated BibTeX entry {match.group(2)}")
        raw = text[match.start() : index]
        body = text[match.end() : index - 1]
        entries.append(
            BibEntry(
                kind=match.group(1).lower(),
                key=match.group(2),
                fields=_parse_fields(body),
                raw=raw,
            )
        )
        cursor = index
    return tuple(entries)


BIB_ENTRIES = parse_bibliography(
    BIB_PATH.read_text(encoding="utf-8")
    + ("\n" + EXTRA_BIB_PATH.read_text(encoding="utf-8") if EXTRA_BIB_PATH.is_file() else "")
)
BIB_BY_KEY = {entry.key: entry for entry in BIB_ENTRIES}


def strip_latex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


_NONQUANTITATIVE_MACROS = (
    "cite",
    "citep",
    "citet",
    "citealp",
    "citeauthor",
    "citeyear",
    "citenum",
    "ref",
    "pageref",
    "autoref",
    "eqref",
    "label",
    "includegraphics",
    "input",
    "include",
    "path",
    "url",
    "href",
    "setcounter",
    "hspace",
    "vspace",
    "setlength",
    "addtolength",
    "fontsize",
)
_NONQUANTITATIVE_MACRO_RE = re.compile(
    r"\\(?:%s)\b\s*(?:\[[^\]]*\])?\s*(?:\{[^{}]*\})?(?:\{[^{}]*\})?"
    % "|".join(_NONQUANTITATIVE_MACROS)
)
_NUMERIC_LITERAL_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_LATEX_LENGTH_RE = re.compile(
    r"^\s*(?:pt|em|ex|bp|dd|pc|sp|cm|mm|in|fil+|\\(?:line|text|column|page|paper)width"
    r"|\\(?:text|paper)height|\\baselineskip|\\height|\\width|\\depth)\b"
)
_HEADING_RE = re.compile(r"\\(section|subsection)\*?\{([^}]+)\}")


def _reader_unit(after: str) -> str | None:
    """Return the direct or shared trailing unit for one numeric occurrence."""

    direct = re.match(
        r"\s*(?:\\?%|(?:(?:-|\\,|~)?\s*(?:kb|Mb|Gb|bp))\b)",
        after,
    )
    if direct:
        return direct.group(0).strip()

    # Scientific prose often writes ``1- and 5-Mb`` or ``4 to 6 Mb``.  The
    # first number carries the same trailing unit even though it is not
    # adjacent to the literal.  Propagate only across numeric list/range
    # syntax, never across ordinary words.
    trailing = re.search(r"(?:(?:-|\\,|~)?\s*(kb|Mb|Gb|bp))\b", after[:80])
    if trailing is None:
        return None
    bridge = after[: trailing.start()]
    # A closing parenthesis or sentence delimiter ends the current numeric
    # range.  Do not leak a later physical unit backward into a confidence
    # interval followed by a separate scale (for example ``0.50--0.74; 2 Mb``).
    if re.search(r"[).;!?]", bridge):
        return None
    bridge = re.sub(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", "", bridge)
    bridge = re.sub(r"\b(?:and|or|to|through)\b", "", bridge, flags=re.IGNORECASE)
    bridge = re.sub(r"[\s,;:/(){}\[\]$~\\-]", "", bridge)
    return trailing.group(1) if not bridge else None


def _manuscript_body(text: str) -> str:
    if r"\begin{document}" in text:
        text = text.split(r"\begin{document}", 1)[1]
    if r"\bibliographystyle" in text:
        text = text.split(r"\bibliographystyle", 1)[0]
    return text


def _reader_text(text: str) -> str:
    """Remove TeX machinery while retaining every printed quantitative literal."""

    text = strip_latex_comments(_manuscript_body(text))
    # A few prose passages use literal references such as ``Fig.~3E--G`` rather
    # than ``\ref``. These are navigation labels, not scientific quantities.
    text = re.sub(
        r"\b(?:main-text\s+)?(?:Fig(?:ure|s)?|Table)\.?~?\s*\d+(?:[A-Za-z](?:--?[A-Za-z])?)?",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\\multicolumn\s*\{\d+\}\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\(?:tfrac|frac)\s*(\d)\s*(\d)", r"\1 \2", text)
    text = re.sub(r"\\renewcommand\s*\{\\arraystretch\}\s*\{[^{}]*\}", " ", text)
    text = _NONQUANTITATIVE_MACRO_RE.sub(" ", text)
    text = re.sub(r"(?<=\d)(?:\{,\}|\\,|,)(?=\d{3}(?:\D|$))", "", text)
    text = text.replace("{=}", "=").replace("{+}", "+").replace("{-}", "-")
    text = text.replace("−", "-")
    text = re.sub(
        r"([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\s*\^\s*\{([+-]?\d+)\}",
        r"\1e\2",
        text,
    )
    # Hyphenated dates and one-hyphen prose ranges contain positive fields;
    # without this normalization, e.g. 2026-07-20 becomes 2026, -07, -20.
    text = re.sub(r"(?<=\d)-(?=\d)", " ", text)
    text = re.sub(
        r"\$([^$]*)\$",
        lambda match: "$"
        + re.sub(r"(?<![A-Za-z_])([+-]?\d+(?:\.\d+)?)(?=[A-Za-z])", r"\1 ", match.group(1))
        + "$",
        text,
    )
    # TeX ranges use two hyphens.  Replace them before tokenization so the
    # upper endpoint in ``0.1--0.5`` cannot be misread as a negative number.
    text = text.replace("---", " ").replace("--", " ")
    text = re.sub(r"(\\[A-Za-z]+)(?=[+-]?\d)", r"\1 ", text)
    return text


def _section_at(text: str, offset: int) -> str:
    section = "front-matter"
    subsection = ""
    for level, title in _HEADING_RE.findall(text[:offset]):
        if level == "section":
            section, subsection = title, ""
        else:
            subsection = title
    return f"{section} > {subsection}" if subsection else section


def reader_numeric_tokens() -> tuple[NumericToken, ...]:
    """Enumerate every reader-visible number in ``main.tex`` and ``si.tex``.

    Cross-reference labels, citations, file paths, URLs, layout dimensions,
    identifiers such as ``F_2``/``2La``/``Mamba-2``, and exponents are not
    quantitative claims and are removed.  Counts, coordinates, method
    parameters, equations, captions, and result values remain.
    """

    tokens: list[NumericToken] = []
    documents: list[tuple[Path, str | None]] = [(MAIN_PATH, None), (SI_PATH, None)]
    documents.extend((path, f"included:{target}") for path, target in included_tex_files())
    seen_paths: set[Path] = set()
    for path, forced_section in documents:
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        raw_body = _manuscript_body(path.read_text(encoding="utf-8"))
        reader = _reader_text(raw_body)
        if path == MAIN_PATH:
            file_id = "main.tex"
        elif path == SI_PATH:
            file_id = "si.tex"
        else:
            file_id = manuscript_id(path.resolve())
        offset = 0
        for line_number, line in enumerate(reader.splitlines(keepends=True), 1):
            for match in _NUMERIC_LITERAL_RE.finditer(line):
                before = line[: match.start()]
                after = line[match.end() :]
                stripped_before = before.rstrip()
                if _LATEX_LENGTH_RE.match(after):
                    continue
                if after[:1].isalpha():
                    continue
                if before[-1:].isalpha() or stripped_before.endswith(("^", "^{", "_", "_{")):
                    continue
                if re.search(r"[A-Za-z][A-Za-z0-9_]*-$", before):
                    continue
                # Preserve enough surrounding prose for semantic provenance
                # ranking (Pearson vs Spearman, P value vs effect size, etc.).
                start = max(0, match.start() - 140)
                end = min(len(line), match.end() + 140)
                context = re.sub(r"\s+", " ", line[start:end]).strip()
                absolute = offset + match.start()
                unit = _reader_unit(after)
                tokens.append(
                    NumericToken(
                        file=file_id,
                        line=line_number,
                        column=match.start() + 1,
                        raw=match.group(0),
                        prefix=re.sub(r"\s+", " ", line[max(0, match.start() - 100) : match.start()]),
                        suffix=re.sub(r"\s+", " ", line[match.end() : match.end() + 100]),
                        context=context,
                        section=forced_section or _section_at(reader, absolute),
                        unit=unit,
                    )
                )
            offset += len(line)
    return tuple(tokens)


def distinct_reader_numbers() -> tuple[tuple[str, str, str, str], ...]:
    """Stable accounting units: ``(file, section, literal, display unit)``."""

    return tuple(
        sorted(
            {
                (token.file, token.section, token.raw, token.unit or "")
                for token in reader_numeric_tokens()
            }
        )
    )


def citation_keys(text: str = MANUSCRIPT) -> tuple[str, ...]:
    keys: set[str] = set()
    for group in re.findall(r"\\cite[A-Za-z*]*\{([^}]*)\}", strip_latex_comments(text)):
        keys.update(key.strip() for key in group.split(",") if key.strip())
    return tuple(sorted(keys))


CITED_KEYS = citation_keys()


def manuscript_urls() -> tuple[str, ...]:
    urls = re.findall(r"\\url\{([^}]+)\}|\\href\{([^}]+)\}", MANUSCRIPT)
    return tuple(sorted({left or right for left, right in urls}))


def provenance_urls() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                source[field]
                for source in PROVENANCE["datasets"]
                for field in ("accession_or_url", "terms_url")
            }
        )
    )


def doi_urls() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"https://doi.org/{entry.fields['doi'].strip()}"
                for entry in BIB_ENTRIES
                if entry.fields.get("doi")
            }
        )
    )


def referenced_graphics() -> tuple[tuple[Path, str], ...]:
    found: list[tuple[Path, str]] = []
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    for tex_path, text in ((MAIN_PATH, MAIN), (SI_PATH, SI)):
        for target in pattern.findall(strip_latex_comments(text)):
            found.append((tex_path.parent / target, target))
    return tuple(found)


def included_tex_files() -> tuple[tuple[Path, str], ...]:
    found: list[tuple[Path, str]] = []
    pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")
    for tex_path, text in ((MAIN_PATH, MAIN), (SI_PATH, SI)):
        for target in pattern.findall(strip_latex_comments(text)):
            path = tex_path.parent / target
            if not path.suffix:
                path = path.with_suffix(".tex")
            found.append((path, target))
    return tuple(found)


def manuscript_file_claims() -> tuple[str, ...]:
    r"""Internal paths exposed through ``\path{...}``, which should remain empty."""

    return tuple(sorted(set(re.findall(r"\\path\{([^}]+)\}", MANUSCRIPT))))


def local_provenance_paths() -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for source in PROVENANCE["datasets"]:
        for field in ("local_derivatives", "producing_scripts"):
            rows.extend((source["id"], field, path) for path in source[field])
    return tuple(rows)


def provenance_citations() -> tuple[tuple[str, str], ...]:
    return tuple(
        (source["id"], key)
        for source in PROVENANCE["datasets"]
        for key in source["citation_keys"]
    )


def result_json_files() -> tuple[Path, ...]:
    roots = (ROOT / "paper" / "results_snapshot", ROOT / "paper" / "figdata")
    return tuple(sorted(path for base in roots for path in base.glob("*.json")))


def result_npz_files() -> tuple[Path, ...]:
    return tuple(sorted((ROOT / "paper" / "figdata").glob("*.npz")))


def producing_python_files() -> tuple[Path, ...]:
    paths = {
        ROOT / path
        for _, field, path in local_provenance_paths()
        if field == "producing_scripts" and path.endswith(".py")
    }
    paths.update((ROOT / "fastrho").glob("*.py"))
    return tuple(sorted(paths))


def producing_shell_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            {
                ROOT / path
                for _, field, path in local_provenance_paths()
                if field == "producing_scripts" and path.endswith(".sh")
            }
        )
    )


def valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and " " not in value


def relative_id(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def manuscript_id(path: Path) -> str:
    """Return a stable id independent of the manuscript checkout location."""

    try:
        return "manuscript/" + str(path.relative_to(MANUSCRIPT_DIR.resolve()))
    except ValueError:
        return relative_id(path)
