#!/usr/bin/env python3
"""Score constant-history pyrho and ReLERNN in exact complete native windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from score_intended_native_multimethod import score_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("raw", "bscorrect"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "history": "constant (deliberately misspecified)",
        "stage": args.stage,
        "complete_windows_only": True,
        "scenarios": {},
    }
    for scenario in ("bottleneck", "expansion"):
        arm = args.root / "arms" / scenario
        result["scenarios"][scenario] = score_config(
            arm,
            arm,
            {"pyrho": arm},
            args.stage,
            True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
