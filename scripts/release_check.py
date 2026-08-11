"""Offline release preflight for software, models, datasets, and manuscript sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANUSCRIPT = pathlib.Path(
    os.environ.get("FASTRHO_MANUSCRIPT_ROOT", ROOT / "tmp" / "reproduce" / "manuscript")
).resolve()


def _load_json(path: pathlib.Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-models", action="store_true",
                        help="fail while any paper model artifact is pending")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    registry = _load_json(ROOT / "fastrho" / "model_registry.json")
    pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation_text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    docs_conf_text = (ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    version_sources = {
        "pyproject.toml": re.search(
            r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']", pyproject_text
        ),
        "CITATION.cff": re.search(r"(?m)^version:\s*[\"']?([^\s\"']+)", citation_text),
        "fastrho/model_registry.json": registry.get("package_version"),
        "docs/conf.py release": re.search(
            r"(?m)^release\s*=\s*[\"']([^\"']+)[\"']", docs_conf_text
        ),
        "docs/conf.py version": re.search(
            r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']", docs_conf_text
        ),
    }
    normalized_versions = {
        source: value.group(1) if hasattr(value, "group") else value
        for source, value in version_sources.items()
    }
    missing_versions = [source for source, value in normalized_versions.items() if not value]
    if missing_versions:
        errors.append(f"missing package version metadata: {', '.join(missing_versions)}")
    elif len(set(normalized_versions.values())) != 1:
        errors.append(f"package version metadata disagrees: {normalized_versions}")
    for model in registry["models"]:
        if model.get("paper_release"):
            for field in ("checkpoint_sha256", "stats_sha256", "model_card", "artifact_manifest"):
                if not model.get(field):
                    errors.append(f"paper model {model['id']} lacks deposit metadata {field}")
            for field in ("checkpoint_sha256", "stats_sha256"):
                value = model.get(field)
                if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                    errors.append(f"paper model {model['id']} has invalid {field}")
            for field in ("model_card", "artifact_manifest"):
                value = model.get(field)
                if value and not (ROOT / value).is_file():
                    errors.append(f"paper model {model['id']} lacks {field}: {value}")
            manifest_name = model.get("artifact_manifest")
            if manifest_name and (ROOT / manifest_name).is_file():
                manifest = _load_json(ROOT / manifest_name)
                if manifest.get("model_id") != model["id"]:
                    errors.append(f"paper model {model['id']} has mismatched artifact manifest")
                for kind, registry_field in (
                    ("checkpoint", "checkpoint_sha256"),
                    ("stats", "stats_sha256"),
                ):
                    recorded = manifest.get("files", {}).get(kind, {}).get("sha256")
                    if recorded != model.get(registry_field):
                        errors.append(
                            f"paper model {model['id']} {registry_field} differs from manifest"
                        )
                training = manifest.get("training", {})
                for field in ("entrypoint", "workflow", "selection_script"):
                    value = training.get(field)
                    if not value or not (ROOT / value).is_file():
                        errors.append(
                            f"paper model {model['id']} lacks reproducible training {field}: {value}"
                        )
                release = manifest.get("release", {})
                for field in ("landing_page", "archive_url", "checkpoint_url", "stats_url"):
                    if model.get("status") == "available" and release.get(field) != model.get(field):
                        errors.append(
                            f"paper model {model['id']} {field} differs between registry and manifest"
                        )
                if model.get("status") == "available" and manifest.get("status") != "available":
                    errors.append(f"paper model {model['id']} manifest is not available")
        if model.get("paper_release") and model.get("status") != "available":
            message = f"paper model {model['id']} has no public artifact"
            (errors if args.strict_models else warnings).append(message)
        if model.get("status") == "available":
            for field in (
                "landing_page", "archive_url", "checkpoint_url", "stats_url",
                "sha256", "archive_sha256",
            ):
                if not model.get(field):
                    errors.append(f"available model {model['id']} lacks {field}")
            for field in ("sha256", "archive_sha256"):
                value = model.get(field)
                if value and not re.fullmatch(r"[0-9a-f]{64}", value):
                    errors.append(f"available model {model['id']} has invalid {field}")
            if model.get("sha256") != model.get("archive_sha256"):
                errors.append(f"available model {model['id']} has inconsistent archive hashes")

    provenance = _load_json(ROOT / "paper" / "data_provenance.yaml")
    required = {"id", "name", "version", "repository", "accession_or_url", "terms_url",
                "citation_keys", "local_derivatives", "producing_scripts", "manuscript_scope"}
    ids = set()
    bibliography_paths = (
        MANUSCRIPT / "refs.bib",
        MANUSCRIPT / "generated_phase2" / "transect_sources.bib",
    )
    bibliography = "\n".join(
        path.read_text(encoding="utf-8") for path in bibliography_paths if path.is_file()
    )
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    main_path = MANUSCRIPT / "main_phase2.tex"
    si_path = MANUSCRIPT / "si_phase2.tex"
    if not main_path.is_file() or not si_path.is_file():
        errors.append(
            "locked Phase 2 manuscript is not staged; run reproduce/fetch_manuscript.py"
        )
    main_text = main_path.read_text(encoding="utf-8") if main_path.is_file() else ""
    si_text = si_path.read_text(encoding="utf-8") if si_path.is_file() else ""
    generated_sources = (
        ROOT / "paper" / "manuscript" / "generated" / "data_sources_table.tex"
    ).read_text(encoding="utf-8")
    manuscript_text = main_text + si_text + generated_sources
    for source in provenance["datasets"]:
        missing = sorted(required - source.keys())
        if missing:
            errors.append(f"dataset {source.get('id', '?')} lacks {missing}")
        if source.get("id") in ids:
            errors.append(f"duplicate dataset id {source['id']}")
        ids.add(source.get("id"))
        for key in source.get("citation_keys", []):
            if key not in bibkeys:
                errors.append(
                    f"dataset {source.get('id')} citation {key} is absent from manuscript bibliographies"
                )
            if key not in manuscript_text:
                errors.append(f"dataset {source.get('id')} citation {key} is absent from manuscript files")
            if source.get("manuscript_scope") == "primary" and key not in main_text:
                errors.append(f"primary dataset {source.get('id')} citation {key} is absent from main text")
        for script in source.get("producing_scripts", []):
            if not (ROOT / script).exists():
                errors.append(f"dataset {source.get('id')} producing script is missing: {script}")
        for derivative in source.get("local_derivatives", []):
            if not (ROOT / derivative).exists():
                errors.append(f"dataset {source.get('id')} derivative is missing: {derivative}")
        if source.get("name") not in generated_sources:
            errors.append(f"dataset {source.get('id')} is absent from generated SI table")

    for path in (ROOT / "README.md", ROOT / "docs"):
        files = [path] if path.is_file() else [
            *path.rglob("*.md"), *path.rglob("*.ipynb"),
        ]
        for file in files:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"/(?:home|Users)/[^\s)`]+", text):
                errors.append(f"host-specific path in public documentation: {file.relative_to(ROOT)}")

    digest = hashlib.sha256((ROOT / "paper" / "data_provenance.yaml").read_bytes()).hexdigest()
    print(f"data provenance sha256: {digest}")
    for message in warnings:
        print(f"WARNING: {message}")
    for message in errors:
        print(f"ERROR: {message}")
    if errors:
        return 1
    print("release metadata checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
