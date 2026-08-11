#!/usr/bin/env python3
"""Verify that every declared public checkpoint asset exists and matches GitHub.

This online audit compares the local user-model and paper-support registries with
GitHub's release metadata.  It checks tag availability, exact asset names, byte
sizes, and GitHub's SHA-256 digest without downloading multi-hundred-MB weights.
The ordinary offline release checks remain network-independent.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.github.com/repos/kevinkorfmann/fastrho-models/releases/tags/{tag}"


def _release_assets(tag: str, timeout: float) -> dict[str, dict[str, object]]:
    request = urllib.request.Request(
        API.format(tag=tag),
        headers={"Accept": "application/vnd.github+json", "User-Agent": "fastrho-release-audit"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"GitHub release {tag!r} returned HTTP {error.code}") from error
    return {asset["name"]: asset for asset in payload.get("assets", [])}


def _check_asset(
    errors: list[str],
    *,
    tag: str,
    assets: dict[str, dict[str, object]],
    name: str,
    expected_sha256: str,
) -> None:
    asset = assets.get(name)
    if asset is None:
        errors.append(f"{tag}: missing asset {name}")
        return
    digest = asset.get("digest")
    if digest != f"sha256:{expected_sha256}":
        errors.append(f"{tag}/{name}: digest {digest!r} != sha256:{expected_sha256}")
    if not isinstance(asset.get("size"), int) or int(asset["size"]) <= 0:
        errors.append(f"{tag}/{name}: invalid remote byte size {asset.get('size')!r}")
    url = str(asset.get("browser_download_url", ""))
    expected_suffix = f"/releases/download/{tag}/{name}"
    if not url.endswith(expected_suffix):
        errors.append(f"{tag}/{name}: unexpected download URL {url!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    errors: list[str] = []
    checked = 0

    registry = json.loads((ROOT / "fastrho" / "model_registry.json").read_text())
    for model in registry["models"]:
        if model.get("status") != "available" or not model.get("public_release"):
            continue
        tag = model["id"]
        assets = _release_assets(tag, args.timeout)
        for name, field in (
            (f"{tag}.zip", "archive_sha256"),
            ("model.ckpt", "checkpoint_sha256"),
            ("feat_stats.npz", "stats_sha256"),
        ):
            _check_asset(
                errors,
                tag=tag,
                assets=assets,
                name=name,
                expected_sha256=model[field],
            )
            checked += 1

    support = json.loads((ROOT / "reproduce" / "checkpoints.json").read_text())
    support_tag = support["paper_support_release"].rstrip("/").rsplit("/", 1)[-1]
    support_assets = _release_assets(support_tag, args.timeout)
    declared_support: dict[str, str] = {}
    for group in support["groups"]:
        for name, digest in group["files"].items():
            previous = declared_support.setdefault(name, digest)
            if previous != digest:
                errors.append(f"paper-support asset {name} has conflicting declared digests")
    for name, digest in declared_support.items():
        _check_asset(
            errors,
            tag=support_tag,
            assets=support_assets,
            name=name,
            expected_sha256=digest,
        )
        checked += 1

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"verified {checked} public checkpoint assets across 7 GitHub releases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
