#!/usr/bin/env python3
"""Verify loose fastrho model files or a complete release archive."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

try:  # package import in tests
    from scripts import package_model_release as release
except ModuleNotFoundError:  # direct ``python scripts/verify_model_release.py`` invocation
    import package_model_release as release


def _manifest(model_id: str) -> dict:
    path = release.ROOT / "models" / model_id / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != model_id:
        raise ValueError(f"manifest model_id is {manifest.get('model_id')!r}")
    return manifest


def verify_archive(model_id: str, archive_path: Path) -> None:
    manifest = _manifest(model_id)
    prefix = f"{model_id}/"
    expected = {
        prefix + "LICENSE",
        prefix + "MODEL_CARD.md",
        prefix + "SHA256SUMS",
        prefix + "feat_stats.npz",
        prefix + "manifest.json",
        prefix + "model.ckpt",
    }
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate member names")
        if set(names) != expected:
            raise ValueError(f"archive members differ: {sorted(set(names) ^ expected)}")
        if archive.testzip() is not None:
            raise ValueError("archive CRC validation failed")
        archived_manifest = json.loads(archive.read(prefix + "manifest.json"))
        repository_manifest = json.loads(
            (release.ROOT / "models" / model_id / "manifest.json").read_text(encoding="utf-8")
        )
        if archived_manifest != repository_manifest:
            raise ValueError("archived manifest differs from the repository manifest")

        checkpoint = archive.read(prefix + "model.ckpt")
        stats = archive.read(prefix + "feat_stats.npz")
        for payload, record, label in (
            (checkpoint, manifest["files"]["checkpoint"], "checkpoint"),
            (stats, manifest["files"]["stats"], "stats"),
        ):
            if len(payload) != record["bytes"]:
                raise ValueError(f"archived {label} has the wrong byte count")
            if release.sha256_bytes(payload) != record["sha256"]:
                raise ValueError(f"archived {label} has the wrong SHA-256")

        listed = {}
        for line in archive.read(prefix + "SHA256SUMS").decode("ascii").splitlines():
            digest, name = line.split("  ", 1)
            listed[name] = digest
        payload_names = expected - {prefix + "SHA256SUMS"}
        if set(listed) != payload_names:
            raise ValueError("SHA256SUMS does not enumerate every payload exactly once")
        for name in payload_names:
            if release.sha256_bytes(archive.read(name)) != listed[name]:
                raise ValueError(f"SHA256SUMS mismatch for {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()

    manifest = _manifest(args.model_id)
    if args.archive:
        if args.stats:
            parser.error("--stats cannot be combined with --archive")
        verify_archive(args.model_id, args.archive)
        print(f"archive_sha256={release.sha256(args.archive)}")
    else:
        if args.stats is None:
            parser.error("--stats is required with --checkpoint")
        release.checked_file(args.checkpoint, manifest["files"]["checkpoint"])
        release.checked_file(args.stats, manifest["files"]["stats"])
        print(f"checkpoint_sha256={release.sha256(args.checkpoint)}")
        print(f"stats_sha256={release.sha256(args.stats)}")
    print("model release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
