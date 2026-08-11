#!/usr/bin/env python3
"""Download, verify, and unpack a model from the fastrho registry."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
import zipfile
from pathlib import Path

try:
    from scripts import package_model_release as release
    from scripts.verify_model_release import verify_archive
except ModuleNotFoundError:
    import package_model_release as release
    from verify_model_release import verify_archive


def registry_record(model_id: str) -> dict:
    registry = json.loads(
        (release.ROOT / "fastrho" / "model_registry.json").read_text(encoding="utf-8")
    )
    matches = [record for record in registry["models"] if record["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"registry contains {len(matches)} records for {model_id!r}")
    record = matches[0]
    if record.get("status") != "available":
        raise ValueError(f"model {model_id!r} is {record.get('status')!r}, not available")
    base = f"https://github.com/kevinkorfmann/fastrho-models/releases/download/{model_id}"
    record.setdefault("archive_url", f"{base}/{model_id}.zip")
    record.setdefault("landing_page", f"https://github.com/kevinkorfmann/fastrho-models/releases/tag/{model_id}")
    return record


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fastrho-model-fetch/1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while block := response.read(8 * 1024 * 1024):
            output.write(block)
    os.replace(temporary, destination)


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(output_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("downloaded-models"))
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    record = registry_record(args.model_id)
    archive_url = record.get("archive_url")
    archive_digest = record.get("archive_sha256") or record.get("sha256")
    if not archive_url or not archive_digest:
        raise ValueError(f"model {args.model_id!r} lacks an archive URL or checksum")
    archive_path = args.output_dir / f"{args.model_id}.zip"
    download(archive_url, archive_path)
    if release.sha256(archive_path) != archive_digest:
        raise ValueError("downloaded archive SHA-256 differs from the registry")
    verify_archive(args.model_id, archive_path)
    safe_extract(archive_path, args.output_dir)
    if not args.keep_archive:
        archive_path.unlink()
    bundle = args.output_dir / args.model_id
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    release.checked_file(bundle / "model.ckpt", manifest["files"]["checkpoint"])
    release.checked_file(bundle / "feat_stats.npz", manifest["files"]["stats"])
    print(f"model_dir={bundle}")
    print("download and verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
