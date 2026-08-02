"""Offline preflight for the public software and model-release metadata."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_json(path: pathlib.Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    errors: list[str] = []
    registry = _load_json(ROOT / "fastrho" / "model_registry.json")

    for model in registry["models"]:
        if model.get("status") != "available":
            continue

        for field in (
            "landing_page",
            "archive_url",
            "checkpoint_url",
            "stats_url",
            "archive_sha256",
            "checkpoint_sha256",
            "stats_sha256",
            "model_card",
            "artifact_manifest",
        ):
            if not model.get(field):
                errors.append(f"available model {model['id']} lacks {field}")

        for field in ("archive_sha256", "checkpoint_sha256", "stats_sha256"):
            value = model.get(field)
            if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                errors.append(f"available model {model['id']} has invalid {field}")

        for field in ("model_card", "artifact_manifest"):
            value = model.get(field)
            if value and not (ROOT / value).is_file():
                errors.append(f"available model {model['id']} lacks {field}: {value}")

        manifest_name = model.get("artifact_manifest")
        if not manifest_name or not (ROOT / manifest_name).is_file():
            continue
        manifest = _load_json(ROOT / manifest_name)
        if manifest.get("model_id") != model["id"]:
            errors.append(f"model {model['id']} has a mismatched artifact manifest")
        for kind, registry_field in (
            ("checkpoint", "checkpoint_sha256"),
            ("stats", "stats_sha256"),
        ):
            recorded = manifest.get("files", {}).get(kind, {}).get("sha256")
            if recorded != model.get(registry_field):
                errors.append(f"model {model['id']} {registry_field} differs from its manifest")
        for field in ("landing_page", "archive_url", "checkpoint_url", "stats_url"):
            if manifest.get("release", {}).get(field) != model.get(field):
                errors.append(f"model {model['id']} {field} differs from its manifest")

    for path in (ROOT / "README.md", ROOT / "docs"):
        files = [path] if path.is_file() else [*path.rglob("*.md"), *path.rglob("*.ipynb")]
        for file in files:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"/(?:home|Users)/[^\s)`]+", text):
                errors.append(f"host-specific path in public documentation: {file.relative_to(ROOT)}")

    registry_hash = hashlib.sha256(
        (ROOT / "fastrho" / "model_registry.json").read_bytes()
    ).hexdigest()
    print(f"model registry sha256: {registry_hash}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        return 1
    print("public release metadata checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
