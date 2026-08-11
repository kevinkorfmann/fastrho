"""Exact, scope-aware provenance for active-manuscript numeric literals."""

from __future__ import annotations

import functools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .inventory import ROOT, NumericToken, reader_numeric_tokens, relative_id

REGISTRY_PATH = ROOT / "paper" / "numeric_provenance.json"
REGISTRY = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ProvenanceMatch:
    token: NumericToken
    source: str
    locator: str
    canonical_value: float
    route: str


def _normalized_source_text(text: str) -> str:
    text = re.sub(r"(?<=\d)(?:\{,\}|\\,|,)(?=\d{3}(?:\D|$))", "", text)
    text = text.replace("−", "-")
    text = re.sub(
        r"([+-]?\d+(?:\.\d+)?)\s*\\times\s*10\s*\^\s*\{([+-]?\d+)\}",
        r"\1e\2",
        text,
    )
    return re.sub(r"(?<=\d)-(?=\d)", " ", text)


_SOURCE_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _string_numbers(value: str, locator: str) -> Iterable[tuple[float, str]]:
    for chromosome in re.finditer(r"\bchr(?:omosome)?[_ -]?(\d+)\b", value, re.IGNORECASE):
        yield float(chromosome.group(1)), f"{locator}:{chromosome.group(0).lower()}"
    for match in _SOURCE_NUMBER.finditer(_normalized_source_text(value)):
        try:
            yield float(match.group(0)), f"{locator}:text"
        except ValueError:
            continue


def _json_numbers(value: object, locator: str = "$") -> Iterable[tuple[float, str]]:
    if isinstance(value, dict):
        chromosome = value.get("chromosome")
        scoped_locator = (
            f"{locator}[chromosome={chromosome}]"
            if isinstance(chromosome, str) and re.fullmatch(r"chr\d+", chromosome, re.IGNORECASE)
            else locator
        )
        for key, child in value.items():
            yield from _string_numbers(str(key), f"{scoped_locator}:key")
            yield from _json_numbers(child, f"{scoped_locator}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _json_numbers(child, f"{locator}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        yield float(value), locator
    elif isinstance(value, str):
        yield from _string_numbers(value, locator)


class _SourceIndex:
    def find_all(
        self, targets: tuple[tuple[float, float, str], ...]
    ) -> tuple[tuple[float, str, str], ...]:
        raise NotImplementedError


class _FlatIndex(_SourceIndex):
    def __init__(self, rows: Iterable[tuple[float, str]]) -> None:
        materialized = [(value, locator) for value, locator in rows if math.isfinite(value)]
        materialized.sort(key=lambda row: row[0])
        self.values = np.asarray([row[0] for row in materialized], dtype=float)
        self.locators = tuple(row[1] for row in materialized)

    def find_all(
        self, targets: tuple[tuple[float, float, str], ...]
    ) -> tuple[tuple[float, str, str], ...]:
        if not self.values.size:
            return ()
        found: list[tuple[float, str, str]] = []
        for target, tolerance, route in targets:
            left = int(np.searchsorted(self.values, target - tolerance, side="left"))
            right = int(np.searchsorted(self.values, target + tolerance, side="right"))
            if left < right:
                indices = sorted(
                    range(left, right), key=lambda i: abs(float(self.values[i]) - target)
                )[:512]
                found.extend(
                    (float(self.values[index]), self.locators[index], route) for index in indices
                )
        return tuple(found)


class _NpzIndex(_SourceIndex):
    def __init__(self, path: Path) -> None:
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                array = np.asarray(archive[key])
                if array.dtype.kind not in "iuf":
                    continue
                flat = array.astype(float, copy=False).ravel()
                keep = np.flatnonzero(np.isfinite(flat))
                if not keep.size:
                    continue
                order = np.argsort(flat[keep], kind="stable")
                arrays[key] = (flat[keep][order], keep[order])
        self.arrays = arrays

    def find_all(
        self, targets: tuple[tuple[float, float, str], ...]
    ) -> tuple[tuple[float, str, str], ...]:
        found: list[tuple[float, str, str]] = []
        for target, tolerance, route in targets:
            for key in sorted(self.arrays):
                values, original = self.arrays[key]
                left = int(np.searchsorted(values, target - tolerance, side="left"))
                right = int(np.searchsorted(values, target + tolerance, side="right"))
                if left < right:
                    indices = sorted(
                        range(left, right), key=lambda i: abs(float(values[i]) - target)
                    )[:16]
                    found.extend(
                        (
                            float(values[index]),
                            f"{key}[flat:{int(original[index])}]",
                            route,
                        )
                        for index in indices
                    )
        return tuple(found)


@functools.lru_cache(maxsize=None)
def source_index(path_string: str) -> _SourceIndex:
    path = Path(path_string)
    if path.suffix == ".npz":
        return _NpzIndex(path)
    if path.suffix in {".json", ".yaml"}:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        else:
            return _FlatIndex(_json_numbers(payload))

    rows: list[tuple[float, str]] = []
    text = _normalized_source_text(path.read_text(encoding="utf-8", errors="replace"))
    for line_number, line in enumerate(text.splitlines(), 1):
        for match in _SOURCE_NUMBER.finditer(line):
            try:
                value = float(match.group(0))
            except ValueError:
                continue
            rows.append((value, f"line:{line_number}"))
    return _FlatIndex(rows)


def _precision(raw: str) -> tuple[float, float]:
    value = float(raw)
    mantissa, exponent = (raw.lower().split("e", 1) + ["0"])[:2]
    decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    # Unqualified integer counts and parameters must be exact. Continuous
    # decimals retain ordinary round-to-nearest tolerance; unit conversions
    # below restore integer rounding where, e.g., a whole number of kb is shown.
    tolerance = (
        1e-12
        if decimals == 0 and "e" not in raw.lower()
        else 0.5 * 10 ** (int(exponent) - decimals) + 1e-12
    )
    return value, tolerance


def target_values(token: NumericToken) -> tuple[tuple[float, float, str], ...]:
    value, tolerance = _precision(token.raw)
    targets = [(value, tolerance, "printed")]
    unit = token.unit or ""
    if "%" in unit:
        percent_tolerance = max(tolerance, 0.5 if "." not in token.raw else tolerance)
        targets.append((value / 100.0, percent_tolerance / 100.0, "percent-to-fraction"))
    for label, multiplier in (("kb", 1_000.0), ("Mb", 1_000_000.0), ("Gb", 1_000_000_000.0)):
        if label in unit:
            unit_tolerance = max(tolerance, 0.5 if "." not in token.raw else tolerance)
            targets.append((value * multiplier, unit_tolerance * multiplier, f"{label}-to-bp"))
    return tuple(targets)


def matching_rule(token: NumericToken) -> dict[str, object] | None:
    matches: list[dict[str, object]] = []
    for rule in REGISTRY["scope_rules"]:
        if rule["file"] != token.file:
            continue
        if "section" in rule and rule["section"] != token.section:
            continue
        if "section_regex" in rule and re.search(str(rule["section_regex"]), token.section) is None:
            continue
        if (
            "context_regex" in rule
            and re.search(str(rule["context_regex"]), token.context, flags=re.IGNORECASE) is None
        ):
            continue
        if (
            "context_exclude_regex" in rule
            and re.search(str(rule["context_exclude_regex"]), token.context, flags=re.IGNORECASE)
            is not None
        ):
            continue
        matches.append(rule)
    if not matches:
        return None
    if len(matches) != 1:
        raise AssertionError(f"ambiguous numeric provenance rules for {token}: {matches}")
    return matches[0]


@functools.lru_cache(maxsize=None)
def group_paths(group: str) -> tuple[Path, ...]:
    patterns = REGISTRY["source_groups"][group]
    found: set[Path] = set()
    for pattern in patterns:
        found.update(path for path in ROOT.glob(pattern) if path.is_file())
    return tuple(
        sorted(found, key=lambda path: (path.suffix in {".py", ".sh", ".sbatch"}, str(path)))
    )


def _derived_constant(token: NumericToken) -> ProvenanceMatch | None:
    value, tolerance = _precision(token.raw)
    for record in REGISTRY["derived_constants"]:
        expected = float(record["value"])
        if abs(value - expected) <= tolerance:
            return ProvenanceMatch(
                token=token,
                source="paper/numeric_provenance.json",
                locator=f"derived_constants:{record['value']}",
                canonical_value=expected,
                route=str(record["kind"]),
            )
    return None


def chromosome_label(token: NumericToken) -> str | None:
    """Return the chromosome named by this literal when it is a label."""

    prefix = token.prefix.lower().replace("~", " ")
    if re.search(r"\bchromosomes?\s+(?:\d+\s*(?:,|and)\s*)*$", prefix[-60:]):
        return token.raw.lstrip("+")
    return None


def nearby_chromosome(token: NumericToken) -> str | None:
    """Return the nearest chromosome label preceding this occurrence."""

    label = chromosome_label(token)
    if label is not None:
        return label
    matches = re.findall(r"\bchromosome\s+([0-9]+)\b", token.prefix.lower()[-100:])
    return matches[-1] if matches else None


def _semantic_penalty(token: NumericToken, source: Path, locator: str) -> int:
    """Prefer a source field whose scientific meaning matches nearby prose."""

    context = token.context.lower()
    near_context = f"{token.prefix[-45:]}{token.raw}{token.suffix[:45]}".lower()
    location = f"{source.name}:{locator}".lower()
    raw = re.escape(token.raw.lower())
    chromosome = nearby_chromosome(token)
    bookkeeping = (
        ("schema_version" in location and "schema" not in context)
        or (".seed" in location and "seed" not in context)
    )
    correlation_context = (
        bool(re.search(rf"(?:correlation|pearson\s+r)\D{{0,25}}{raw}", context))
        and "spearman" not in near_context
        and "log-pearson" not in near_context
    )
    desired: list[tuple[bool, tuple[str, ...]]] = [
        (
            correlation_context
            or bool(re.search(rf"(?:pearson|pearson\s+correlations?).{{0,90}}{raw}", context)),
            ("pearson",),
        ),
        (bool(re.search(rf"(?:spearman|r_s\s*=).{{0,90}}{raw}", context)), ("spearman",)),
        (
            bool(re.search(rf"p\s*=\s*{raw}", context)),
            (
                "p_one_sided",
                "p_two_sided",
                "p_value",
                "pvalue",
                "permutation_p",
                "shift_p",
                "pearson_h",
                "$.p",
            ),
        ),
        (
            bool(re.search(rf"(?:interval|range).{{0,45}}{raw}", context)),
            ("ci", "interval", "bootstrap", "range"),
        ),
        (bool(re.search(rf"n\s*=\s*{raw}", context)), (".n", "_n", "count")),
        (bool(re.search(rf"fastrho.{{0,100}}{raw}", context)), ("fastrho",)),
        (bool(re.search(rf"pyrho.{{0,100}}{raw}", context)), ("pyrho",)),
        (bool(re.search(rf"relernn.{{0,100}}{raw}", context)), ("relernn",)),
        ("cohort" in near_context, ("cohort",)),
        ("accession" in near_context, ("accession", "sample", "count")),
        ("haplotype" in near_context and token.raw.isdigit(), ("hap", "sample", "count")),
        ("bias ratio" in context or "estimated-to-true" in context, ("bias_ratio",)),
        (bool(re.search(rf"{raw}.{{0,15}}fold", context)), ("ratio", "fold")),
        ("coverage" in near_context and "%" in (token.unit or ""), ("coverage",)),
        (
            "genotype error probability" in near_context
            or "genotype-error probability" in near_context,
            ("genotype_error_probability",),
        ),
        (
            any(label in near_context for label in ("runtime", "wall-clock", "cost of")),
            ("timing", "runtime", "elapsed"),
        ),
    ]
    penalty = 3 * int(":key" in locator) + 3 * int(locator.endswith(":text")) + sum(
        2 * int(active and not any(label in location for label in labels))
        for active, labels in desired
    )
    if chromosome is not None and f"chr{chromosome}" not in location:
        penalty += 2
    # Results and Discussion claims should resolve to result artifacts whenever
    # possible. Executable source remains the right authority for Methods
    # parameters, but an incidental literal in plotting code is weak evidence
    # for a reported empirical value.
    if ("results" in token.section.lower() or token.section == "Discussion") and source.suffix in {
        ".py",
        ".sh",
        ".sbatch",
    }:
        penalty += 2
    if correlation_context and "log_pearson" in location:
        penalty += 1
    unit = (token.unit or "").lower()
    if "kb" in unit and not any(label in location for label in ("kb", "bp", "window", "distance")):
        penalty += 1
    if "mb" in unit and not any(label in location for label in ("mb", "bp", "position", "distance")):
        penalty += 1
    if "gb" in unit and not any(label in location for label in ("gb", "bp", "length")):
        penalty += 1
    if source.name.startswith("slim_") and not any(
        word in context for word in ("slim", "sweep", "background selection", "neutral")
    ):
        penalty += 1
    if any(label in location for label in ("sha256", "checksum")) and not any(
        label in context for label in ("sha256", "checksum", "hash")
    ):
        penalty += 16
    if bookkeeping:
        penalty += 8
    return penalty


def _unit_route_penalty(token: NumericToken, locator: str, route: str) -> int:
    """Prefer a field with compatible units over an unrelated equal literal."""

    unit = (token.unit or "").lower()
    if not unit:
        return 0
    location = locator.lower()
    for label in ("kb", "mb", "gb"):
        if label not in unit:
            continue
        if route == f"{label.capitalize() if label != 'kb' else 'kb'}-to-bp":
            return 0
        if any(marker in location for marker in ("_bp", "position", "distance", "window", "gap", "start", "end", "length", "scale", "edges")):
            return 0
        return 2
    return 0


def find_provenance(token: NumericToken) -> ProvenanceMatch | None:
    rule = matching_rule(token)
    if rule is None:
        return _derived_constant(token)
    targets = target_values(token)
    candidates: list[tuple[int, int, float, int, int, Path, float, str, str]] = []
    target_by_route = {route: (target, tolerance) for target, tolerance, route in targets}
    for group_index, group in enumerate(rule["groups"]):
        for path_index, path in enumerate(group_paths(str(group))):
            for value, locator, route in source_index(str(path)).find_all(targets):
                target, tolerance = target_by_route[route]
                scaled_error = abs(value - target) / max(tolerance, 1e-15)
                candidates.append(
                    (
                        _unit_route_penalty(token, locator, route),
                        _semantic_penalty(token, path, locator),
                        scaled_error,
                        group_index,
                        path_index,
                        path,
                        value,
                        locator,
                        route,
                    )
                )
    if candidates:
        _, _, _, _, _, path, value, locator, route = min(candidates)
        return ProvenanceMatch(
            token=token,
            source=relative_id(path),
            locator=locator,
            canonical_value=value,
            route=route,
        )
    return _derived_constant(token)


def provenance_rows() -> tuple[ProvenanceMatch | None, ...]:
    """One deterministic result per distinct file/section/value/unit claim."""

    representatives: dict[tuple[str, str, str, str], NumericToken] = {}
    for token in reader_numeric_tokens():
        key = (token.file, token.section, token.raw, token.unit or "")
        representatives.setdefault(key, token)
    return tuple(find_provenance(representatives[key]) for key in sorted(representatives))
