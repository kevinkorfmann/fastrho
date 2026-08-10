#!/usr/bin/env python3
"""Run a manuscript species preset on a prepared VCF or genotype-matrix NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "species.json"


def load_species(key: str) -> dict:
    with MANIFEST.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    matches = [row for row in manifest["species"] if row["key"] == key]
    if not matches:
        available = ", ".join(sorted(row["key"] for row in manifest["species"]))
        raise SystemExit(f"unknown species preset {key!r}; choose from: {available}")
    return matches[0]


def command_preview(args: argparse.Namespace, preset: dict, source: Path) -> str:
    inference = preset["inference"]
    parts = [
        "python",
        str(Path(__file__).name),
        "--species",
        preset["key"],
        "--vcf" if args.vcf else "--npz",
        str(source),
        "--checkpoint",
        str(args.checkpoint),
        "--stats",
        str(args.stats),
        "--out",
        str(args.out),
        "--device",
        args.device,
        "--input-mode",
        args.input_mode or inference["input_mode"],
        "--window-size",
        str(args.window_size or inference["window_size_bp"]),
    ]
    if args.contig:
        parts.extend(["--contig", args.contig])
    if args.ne is not None:
        parts.extend(["--ne", str(args.ne)])
    return " ".join(shlex.quote(part) for part in parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--species", required=True, help="key from species.json or data.py list")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--vcf", type=Path, help="complete, biallelic, single-cohort VCF")
    source.add_argument("--npz", type=Path, help="NPZ containing gm[n_hap,n_snp] and pos[n_snp]")
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--stats", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--contig", help="override the preset contig for a VCF")
    result.add_argument("--mutation-rate", type=float, help="override the documented preset value")
    result.add_argument("--input-mode", choices=["phased", "unphased", "unpolarized"])
    result.add_argument("--ne", type=float, help="fixed diploid Ne; otherwise fastrho estimates it")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--window-size", type=int, help="BED window size in bp")
    result.add_argument("--dry-run", action="store_true", help="print resolved settings only")
    return result


def main() -> None:
    args = parser().parse_args()
    preset = load_species(args.species)
    inference = preset["inference"]
    source = args.vcf or args.npz
    contig = args.contig or inference["contig"]
    mutation_rate = args.mutation_rate or inference["mutation_rate"]
    input_mode = args.input_mode or inference["input_mode"]
    window_size = args.window_size or inference["window_size_bp"]

    print(f"species:       {preset['common_name']} ({preset['scientific_name']})")
    print(f"source:        {source}")
    print(f"contig:        {contig}")
    print(f"mutation rate: {mutation_rate:.4g}")
    print(f"input mode:    {input_mode}")
    print(f"model profile: {inference['model_profile']}")
    print(f"BED window:    {window_size} bp")
    print(f"command:       {command_preview(args, preset, source)}")
    if args.dry_run:
        return

    import fastrho

    if args.npz:
        import numpy as np

        archive = np.load(args.npz)
        if not {"gm", "pos"}.issubset(archive.files):
            raise SystemExit("NPZ must contain gm[n_hap,n_snp] and pos[n_snp]")
        model, cfg, stats = fastrho.load_model(
            str(args.checkpoint), str(args.stats), device=args.device
        )
        prediction = fastrho.predict_map_from_genotype_matrix(
            archive["gm"],
            archive["pos"],
            model,
            cfg,
            stats,
            mutation_rate=mutation_rate,
            Ne=args.ne,
            device=args.device,
            input_mode=input_mode,
        )
    else:
        prediction = fastrho.quick_map_from_vcf(
            str(args.vcf),
            str(args.checkpoint),
            str(args.stats),
            contig=contig,
            mutation_rate=mutation_rate,
            Ne=args.ne,
            device=args.device,
            input_mode=input_mode,
            missing="drop-site",
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fastrho.write_bed(prediction, str(args.out), chrom=contig, window_size=window_size)
    print(f"wrote:         {args.out}")
    record = {
        "schema_version": 1,
        "fastrho_version": fastrho.__version__,
        "species_preset": preset["key"],
        "scientific_name": preset["scientific_name"],
        "input": str(source),
        "input_kind": "vcf" if args.vcf else "npz",
        "contig": contig,
        "mutation_rate": mutation_rate,
        "input_mode": input_mode,
        "Ne": args.ne,
        "model_id": inference["model_id"],
        "model_profile": inference["model_profile"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "stats": str(args.stats),
        "stats_sha256": sha256(args.stats),
        "device": args.device,
        "window_size_bp": window_size,
        "output": str(args.out),
        "command": command_preview(args, preset, source),
    }
    provenance = args.out.with_name(args.out.name + ".provenance.json")
    provenance.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"provenance:    {provenance}")


if __name__ == "__main__":
    main()
