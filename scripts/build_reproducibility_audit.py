#!/usr/bin/env python3
"""Build the deterministic manuscript number-and-figure audit artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.verification.inventory import reader_numeric_tokens  # noqa: E402
from tests.verification.numeric_provenance import find_provenance  # noqa: E402

OUTPUT = ROOT / "paper" / "reproducibility_audit.json"
FIGURES = ROOT / "paper" / "figure_provenance.json"


def build_audit() -> dict[str, object]:
    """Return one exact source binding per printed occurrence and included figure."""

    occurrences = reader_numeric_tokens()
    numbers = []
    for occurrence_index, token in enumerate(occurrences, 1):
        match = find_provenance(token)
        numbers.append(
            {
                "occurrence_id": f"{token.file}:{token.line}:{token.column}:{occurrence_index}",
                "manuscript_file": token.file,
                "section": token.section,
                "line": token.line,
                "column": token.column,
                "printed": token.raw,
                "unit": token.unit,
                "context": token.context,
                "resolved": match is not None,
                "source": match.source if match else None,
                "source_locator": match.locator if match else None,
                "canonical_value": match.canonical_value if match else None,
                "conversion": match.route if match else None,
            }
        )

    figure_manifest = json.loads(FIGURES.read_text(encoding="utf-8"))
    unresolved = sum(not row["resolved"] for row in numbers)
    distinct = {
        (token.file, token.section, token.raw, token.unit or "") for token in occurrences
    }
    return {
        "schema_version": 2,
        "policy": (
            "Every reader-facing numeric occurrence must resolve to a scoped, "
            "committed source; every included figure must resolve to one executable "
            "producer with checksummed inputs and output."
        ),
        "summary": {
            "numeric_occurrences": len(occurrences),
            "distinct_numeric_claims": len(distinct),
            "resolved_numeric_occurrences": len(numbers) - unresolved,
            "unresolved_numeric_occurrences": unresolved,
            "figures": len(figure_manifest["figures"]),
        },
        "numbers": numbers,
        "figures": figure_manifest["figures"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    serialized = json.dumps(build_audit(), indent=2) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"stale reproducibility audit: run {Path(__file__).name}")
    else:
        OUTPUT.write_text(serialized, encoding="utf-8")
        print(OUTPUT)


if __name__ == "__main__":
    main()
