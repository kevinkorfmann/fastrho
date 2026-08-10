#!/usr/bin/env python3
"""Fetch and verify the exact locked Phase 2 manuscript snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = Path(__file__).with_name("manuscript.lock.json")
DEFAULT_DESTINATION = ROOT / "tmp" / "reproduce" / "manuscript"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_lock() -> dict[str, object]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("variant") != "phase2":
        raise ValueError("the manuscript lock must select Phase 2")
    return lock


def verify(destination: Path, lock: dict[str, object]) -> None:
    errors = []
    for relative, expected in lock["files"].items():
        path = destination / relative
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            errors.append(f"checksum mismatch: {relative} ({observed})")
    if errors:
        raise RuntimeError("Phase 2 manuscript verification failed:\n- " + "\n- ".join(errors))


def from_checkout(source: Path, destination: Path, lock: dict[str, object]) -> None:
    verify(source, lock)
    for relative in lock["files"]:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)


def from_github(destination: Path, lock: dict[str, object]) -> None:
    repository = lock["repository"]
    commit = lock["commit"]
    url = f"https://api.github.com/repos/{repository}/zipball/{commit}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "fastrho-reproducibility",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token and shutil.which("gh"):
        result = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            token = result.stdout.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:  # noqa: S310 - pinned GitHub URL
        archive_bytes = response.read()
    archive_path = destination.parent / f"manuscript-{commit}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(archive_bytes)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = {PurePosixPath(name): name for name in archive.namelist()}
            prefixes = {path.parts[0] for path in members if path.parts}
            if len(prefixes) != 1:
                raise RuntimeError("unexpected GitHub archive layout")
            prefix = next(iter(prefixes))
            for relative, expected in lock["files"].items():
                member = PurePosixPath(prefix) / relative
                member_name = members.get(member)
                if member_name is None:
                    raise RuntimeError(f"locked manuscript file missing from archive: {relative}")
                data = archive.read(member_name)
                if sha256(data) != expected:
                    raise RuntimeError(f"locked manuscript checksum mismatch: {relative}")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
    finally:
        archive_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--source", type=Path, help="use a clean local manuscript checkout")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    lock = load_lock()
    destination = args.destination.expanduser().resolve()
    if not args.verify_only:
        if args.source:
            from_checkout(args.source.expanduser().resolve(), destination, lock)
        else:
            from_github(destination, lock)
    verify(destination, lock)
    print(f"verified Phase 2 manuscript {lock['commit']} at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
