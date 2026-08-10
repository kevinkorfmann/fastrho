#!/usr/bin/env python3
"""Inspect, download, and prepare the manuscript species datasets.

The utility is intentionally standard-library-only. It never starts a download unless the
``download`` subcommand is used, writes through a ``.part`` file, and refuses to overwrite files
unless ``--force`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "species.json"


def load_manifest() -> dict:
    with MANIFEST.open(encoding="utf-8") as handle:
        return json.load(handle)


def presets(manifest: dict) -> dict[str, dict]:
    return {entry["key"]: entry for entry in manifest["species"]}


def human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown size"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def get_preset(args: argparse.Namespace, manifest: dict) -> dict:
    try:
        return presets(manifest)[args.species]
    except KeyError as exc:
        available = ", ".join(sorted(presets(manifest)))
        raise SystemExit(
            f"unknown species preset {args.species!r}; choose from: {available}"
        ) from exc


def command_list(args: argparse.Namespace, manifest: dict) -> None:
    rows = manifest["species"]
    if args.role:
        rows = [row for row in rows if args.role in row["paper_roles"]]
    print("key\tcommon name\tscientific name\trole\taccess\tcontig")
    for row in rows:
        print(
            "\t".join(
                [
                    row["key"],
                    row["common_name"],
                    row["scientific_name"],
                    ",".join(row["paper_roles"]),
                    row["source"]["download_kind"],
                    row["inference"]["contig"],
                ]
            )
        )


def command_show(args: argparse.Namespace, manifest: dict) -> None:
    row = get_preset(args, manifest)
    source = row["source"]
    inference = row["inference"]
    cohort = row["cohort"]
    cohort_size = (
        f"{cohort['n_individuals']} individuals"
        if cohort["n_individuals"] is not None
        else "cohort-defined sample count"
    )
    print(f"{row['common_name']} ({row['scientific_name']})")
    print(f"preset:        {row['key']}")
    print(f"paper role:    {', '.join(row['paper_roles'])}")
    print(f"cohort:        {cohort_size}; {cohort['selection']}")
    print(f"source:        {source['label']}")
    print(f"source page:   {source['landing_page']}")
    print(f"access route:  {source['download_kind']}")
    for file in source.get("files", []):
        suffix = f" ({human_bytes(file.get('size_bytes'))})"
        print(f"  {file['role']}: {file['filename']}{suffix}")
        print(f"    {file['url']}")
    print(f"contig:        {inference['contig']}")
    print(f"mutation rate: {inference['mutation_rate']:.4g} per site per generation")
    print(f"input mode:    {inference['input_mode']}")
    print(f"model:         {inference['model_profile']}")
    print(f"window size:   {inference['window_size_bp']} bp")
    if row.get("notes"):
        print(f"notes:         {row['notes']}")


def checksum(path: Path, specification: str) -> bool:
    algorithm, expected = specification.split(":", 1)
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower() == expected.lower()


def download_one(file: dict, outdir: Path, *, force: bool, dry_run: bool) -> None:
    target = outdir / file["filename"]
    partial = target.with_name(target.name + ".part")
    print(f"{file['url']}\n  -> {target} ({human_bytes(file.get('size_bytes'))})")
    if dry_run:
        return
    if target.exists() and not force:
        raise SystemExit(f"refusing to overwrite {target}; pass --force to replace it")
    outdir.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(file["url"], headers={"User-Agent": "fastrho-data/1"})
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if file.get("checksum") and not checksum(partial, file["checksum"]):
            raise RuntimeError(f"checksum mismatch for {file['filename']}")
        os.replace(partial, target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def command_download(args: argparse.Namespace, manifest: dict) -> None:
    row = get_preset(args, manifest)
    source = row["source"]
    if source["download_kind"] != "direct":
        print(f"{row['key']} uses the {source['download_kind']} access route.")
        print(f"Open {source['landing_page']}")
        if source.get("instructions"):
            print(source["instructions"])
        raise SystemExit(0 if args.dry_run else 2)
    files = source.get("files", [])
    selected = (
        files if args.include_companions else [file for file in files if file["role"] == "primary"]
    )
    if not selected:
        raise SystemExit(f"no downloadable primary file is registered for {row['key']}")
    for file in selected:
        download_one(file, args.outdir, force=args.force, dry_run=args.dry_run)


def shell_join(parts: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in parts)


def command_prepare(args: argparse.Namespace, manifest: dict) -> None:
    row = get_preset(args, manifest)
    contig = args.contig or row["inference"]["contig"]
    region = contig
    if args.start or args.end:
        region += f":{args.start or 1}-{args.end or ''}"
    command = [
        args.bcftools,
        "view",
        "--threads",
        str(args.threads),
        "-t",
        region,
        "-m2",
        "-M2",
        "-v",
        "snps",
        "-e",
        'GT="mis"',
    ]
    if args.samples:
        command.extend(["-S", str(args.samples)])
    command.extend(["-Oz", "-o", str(args.out), str(args.vcf)])
    index_command = [args.bcftools, "index", "-t", "--force", str(args.out)]
    print(shell_join(command))
    print(shell_join(index_command))
    if args.dry_run:
        return
    if shutil.which(args.bcftools) is None:
        raise SystemExit(f"{args.bcftools!r} was not found; install bcftools or use --dry-run")
    if args.out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.out}; pass --force to replace it")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    subprocess.run(index_command, check=True)
    record = {
        "schema_version": 1,
        "species_preset": row["key"],
        "manifest_accessed": manifest["accessed"],
        "source_page": row["source"]["landing_page"],
        "source_vcf": str(args.vcf),
        "region": region,
        "sample_file": str(args.samples) if args.samples else None,
        "sample_ids": args.samples.read_text(encoding="utf-8").split() if args.samples else None,
        "command": command,
        "output": str(args.out),
    }
    provenance = args.out.with_name(args.out.name + ".provenance.json")
    provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {provenance}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list", help="list the manuscript species presets")
    list_parser.add_argument(
        "--role", choices=["transect", "primary_anopheles", "primary_redpoll", "canid_control"]
    )
    list_parser.set_defaults(function=command_list)

    show_parser = sub.add_parser("show", help="show source, cohort, and inference metadata")
    show_parser.add_argument("species")
    show_parser.set_defaults(function=command_show)

    download_parser = sub.add_parser("download", help="download a registered source file")
    download_parser.add_argument("species")
    download_parser.add_argument("--outdir", type=Path, default=HERE / "data")
    download_parser.add_argument("--include-companions", action="store_true")
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--dry-run", action="store_true")
    download_parser.set_defaults(function=command_download)

    prepare_parser = sub.add_parser("prepare", help="make a complete, biallelic, one-contig VCF")
    prepare_parser.add_argument("species")
    prepare_parser.add_argument("--vcf", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    prepare_parser.add_argument("--samples", type=Path, help="one VCF sample ID per line")
    prepare_parser.add_argument(
        "--contig", help="override the manifest contig after checking the header"
    )
    prepare_parser.add_argument("--start", type=int)
    prepare_parser.add_argument("--end", type=int)
    prepare_parser.add_argument("--threads", type=int, default=4)
    prepare_parser.add_argument("--bcftools", default="bcftools")
    prepare_parser.add_argument("--force", action="store_true")
    prepare_parser.add_argument("--dry-run", action="store_true")
    prepare_parser.set_defaults(function=command_prepare)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args, load_manifest())


if __name__ == "__main__":
    main()
