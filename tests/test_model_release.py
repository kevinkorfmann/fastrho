from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from fastrho import model_release as installed_release
from scripts import package_model_release as release
from scripts import verify_model_release as verifier


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fixture_root(tmp_path: Path, checkpoint: bytes = b"checkpoint", stats: bytes = b"stats"):
    model_dir = tmp_path / "models" / "test-v1"
    model_dir.mkdir(parents=True)
    manifest = {
        "model_id": "test-v1",
        "files": {
            "checkpoint": {
                "name": "model.ckpt",
                "bytes": len(checkpoint),
                "sha256": digest(checkpoint),
            },
            "stats": {
                "name": "feat_stats.npz",
                "bytes": len(stats),
                "sha256": digest(stats),
            },
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (model_dir / "MODEL_CARD.md").write_text("# Test model\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("test license\n", encoding="utf-8")
    checkpoint_path = tmp_path / "model.ckpt"
    stats_path = tmp_path / "feat_stats.npz"
    checkpoint_path.write_bytes(checkpoint)
    stats_path.write_bytes(stats)
    return checkpoint_path, stats_path


def build(monkeypatch, tmp_path: Path, output: Path) -> None:
    checkpoint, stats = fixture_root(tmp_path)
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package_model_release.py",
            "--model-id",
            "test-v1",
            "--checkpoint",
            str(checkpoint),
            "--stats",
            str(stats),
            "--output",
            str(output),
        ],
    )
    assert release.main() == 0


def test_model_release_archive_is_deterministic(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = second_root / "second.zip"
    build(monkeypatch, tmp_path, first)
    build(monkeypatch, second_root, second)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == sorted(archive.namelist())
        assert "test-v1/model.ckpt" in archive.namelist()
        assert "test-v1/feat_stats.npz" in archive.namelist()
        assert "test-v1/SHA256SUMS" in archive.namelist()
    verifier.verify_archive("test-v1", first)


def test_model_release_rejects_wrong_checkpoint(tmp_path: Path) -> None:
    checkpoint, _ = fixture_root(tmp_path)
    checkpoint.write_bytes(b"different")
    manifest = json.loads(
        (tmp_path / "models" / "test-v1" / "manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="bytes; expected"):
        release.checked_file(checkpoint, manifest["files"]["checkpoint"])


def test_installed_release_verifies_archive(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "release.zip"
    build(monkeypatch, tmp_path, output)
    record = {
        "checkpoint_sha256": digest(b"checkpoint"),
        "stats_sha256": digest(b"stats"),
    }
    installed_release.verify_archive("test-v1", output, record)


def test_installed_release_safe_extract_rejects_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../outside", b"unsafe")
    with pytest.raises(ValueError, match="unsafe archive member"):
        installed_release.safe_extract(archive_path, tmp_path / "models")


def test_archive_verification_rejects_modified_payload(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "release.zip"
    build(monkeypatch, tmp_path, output)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(output, "a") as archive:
            archive.writestr("test-v1/model.ckpt", b"tampered")
    with pytest.raises(ValueError, match="duplicate member"):
        verifier.verify_archive("test-v1", output)


def test_primary_model_manifest_is_deposit_ready() -> None:
    manifest = json.loads(
        (release.ROOT / "models" / "domain-randomized-v1" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "available"
    assert manifest["files"]["checkpoint"]["bytes"] == 196930253
    assert len(manifest["files"]["checkpoint"]["sha256"]) == 64
    assert len(manifest["files"]["stats"]["sha256"]) == 64
    assert manifest["training"]["selected_epoch"] == 53
    assert manifest["training"]["validation_pearson"] == pytest.approx(0.8618586659431458)
    assert manifest["release"]["landing_page"].startswith("https://github.com/")
    assert manifest["release"]["archive_url"].endswith("domain-randomized-v1.zip")
