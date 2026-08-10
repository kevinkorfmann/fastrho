#!/usr/bin/env python3
"""Stage regenerated paper artifacts into a private Phase 2 build tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(__file__).with_name("artifacts.json")
LOCK = Path(__file__).with_name("manuscript.lock.json")
DEFAULT_MANUSCRIPT = ROOT / "tmp" / "reproduce" / "manuscript"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument(
        "--check-published",
        action="store_true",
        help="require every package artifact to match the published manuscript byte-for-byte",
    )
    args = parser.parse_args()
    manuscript = args.manuscript.expanduser().resolve()
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    registry = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    errors = []
    copies: list[tuple[Path, Path]] = []
    for artifact in registry["artifacts"]:
        source = ROOT / artifact["package_output"]
        target = manuscript / artifact["manuscript_target"]
        if not source.is_file():
            errors.append(f"missing package artifact: {artifact['package_output']}")
            continue
        expected = lock["files"].get(artifact["manuscript_target"])
        if args.check_published and expected != digest(source):
            errors.append(
                f"does not match published Phase 2: {artifact['package_output']} -> "
                f"{artifact['manuscript_target']}"
            )
            continue
        copies.append((source, target))
    if errors:
        raise SystemExit("artifact staging failed:\n- " + "\n- ".join(errors))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f"staged {len(registry['artifacts'])} Phase 2 artifacts in {manuscript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
