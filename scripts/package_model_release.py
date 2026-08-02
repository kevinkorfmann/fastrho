"""Build a deterministic, checksummed pretrained-model deposit archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(path: Path, record: dict[str, object]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != record["bytes"]:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; expected {record['bytes']}"
        )
    observed = sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{path} has SHA-256 {observed}; expected {record['sha256']}")


def write_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload, compresslevel=9)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def release_members(model_id: str, checkpoint: Path, stats: Path) -> dict[str, bytes]:
    """Return the complete, ordered-input payload for a model release archive."""
    model_dir = ROOT / "models" / model_id
    manifest_path = model_dir / "manifest.json"
    model_card_path = model_dir / "MODEL_CARD.md"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["model_id"] != model_id:
        raise ValueError(f"manifest model_id is {manifest['model_id']!r}")

    checked_file(checkpoint, manifest["files"]["checkpoint"])
    checked_file(stats, manifest["files"]["stats"])
    members = {
        f"{model_id}/model.ckpt": checkpoint.read_bytes(),
        f"{model_id}/feat_stats.npz": stats.read_bytes(),
        f"{model_id}/MODEL_CARD.md": model_card_path.read_bytes(),
        f"{model_id}/manifest.json": manifest_path.read_bytes(),
        f"{model_id}/LICENSE": (ROOT / "LICENSE").read_bytes(),
    }
    checksum_lines = [
        f"{sha256_bytes(payload)}  {name}\n"
        for name, payload in sorted(members.items())
    ]
    members[f"{model_id}/SHA256SUMS"] = "".join(checksum_lines).encode("ascii")
    return members


def build_archive(model_id: str, checkpoint: Path, stats: Path, output: Path) -> None:
    members = release_members(model_id, checkpoint, stats)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name, payload in sorted(members.items()):
            write_member(archive, name, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    build_archive(args.model_id, args.checkpoint, args.stats, args.output)

    print(f"archive={args.output}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
