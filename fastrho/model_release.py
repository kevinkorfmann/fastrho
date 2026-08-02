"""Download and verify a pretrained model from the packaged fastrho registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from importlib.resources import files
from pathlib import Path


def _registry() -> dict:
    resource = files("fastrho").joinpath("model_registry.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def registry_record(model_id: str) -> dict:
    matches = [record for record in _registry()["models"] if record["id"] == model_id]
    if len(matches) != 1:
        raise ValueError(f"registry contains {len(matches)} records for {model_id!r}")
    record = matches[0]
    if record.get("status") != "available":
        raise ValueError(f"model {model_id!r} is {record.get('status')!r}, not available")
    return record


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "fastrho-model-fetch/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_archive(model_id: str, archive_path: Path, record: dict | None = None) -> None:
    record = registry_record(model_id) if record is None else record
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

        manifest = json.loads(archive.read(prefix + "manifest.json"))
        if manifest.get("model_id") != model_id:
            raise ValueError(f"manifest model_id is {manifest.get('model_id')!r}")

        for kind, registry_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("stats", "stats_sha256"),
        ):
            file_record = manifest["files"][kind]
            payload = archive.read(prefix + file_record["name"])
            if len(payload) != file_record["bytes"]:
                raise ValueError(f"archived {kind} has the wrong byte count")
            observed = sha256_bytes(payload)
            if observed != file_record["sha256"] or observed != record[registry_field]:
                raise ValueError(f"archived {kind} has the wrong SHA-256")

        listed = {}
        for line in archive.read(prefix + "SHA256SUMS").decode("ascii").splitlines():
            digest, name = line.split("  ", 1)
            listed[name] = digest
        payload_names = expected - {prefix + "SHA256SUMS"}
        if set(listed) != payload_names:
            raise ValueError("SHA256SUMS does not enumerate every payload exactly once")
        for name in payload_names:
            if sha256_bytes(archive.read(name)) != listed[name]:
                raise ValueError(f"SHA256SUMS mismatch for {name}")


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe archive member: {member.filename}")
        archive.extractall(output_dir)


def _check_extracted(bundle: Path, record: dict) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    for kind, registry_field in (
        ("checkpoint", "checkpoint_sha256"),
        ("stats", "stats_sha256"),
    ):
        file_record = manifest["files"][kind]
        path = bundle / file_record["name"]
        if path.stat().st_size != file_record["bytes"]:
            raise ValueError(f"{path} has the wrong byte count")
        observed = sha256_path(path)
        if observed != file_record["sha256"] or observed != record[registry_field]:
            raise ValueError(f"{path} has the wrong SHA-256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="domain-randomized-v1")
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
    if sha256_path(archive_path) != archive_digest:
        raise ValueError("downloaded archive SHA-256 differs from the registry")
    verify_archive(args.model_id, archive_path, record)
    safe_extract(archive_path, args.output_dir)
    if not args.keep_archive:
        archive_path.unlink()

    bundle = args.output_dir / args.model_id
    _check_extracted(bundle, record)
    print(f"model_dir={bundle}")
    print("download and verification passed")
    return 0


def verify_main() -> int:
    parser = argparse.ArgumentParser(description="Verify a downloaded fastrho model bundle.")
    parser.add_argument("--model-id", default="domain-randomized-v1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--bundle", type=Path)
    args = parser.parse_args()

    record = registry_record(args.model_id)
    if args.archive:
        expected = record.get("archive_sha256") or record.get("sha256")
        if not expected or sha256_path(args.archive) != expected:
            raise ValueError("archive SHA-256 differs from the registry")
        verify_archive(args.model_id, args.archive, record)
        print(f"archive_sha256={sha256_path(args.archive)}")
    else:
        _check_extracted(args.bundle, record)
        print(f"model_dir={args.bundle}")
    print("model release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
