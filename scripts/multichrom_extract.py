"""Extract several real chromosomes in one VCF pass for manuscript robustness tests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from cyvcf2 import VCF


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--chrom", action="append", required=True)
    parser.add_argument("--mode", choices=("dosage", "haploid"), default="dosage")
    parser.add_argument("--samples", default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-missing", type=float, default=0.1)
    parser.add_argument("--min-maf", type=float, default=0.0)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--mu", type=float, required=True)
    parser.add_argument("--map-sp", default="")
    parser.add_argument("--map-id", default="")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    requested = set(args.chrom)
    selected = None
    if args.samples:
        selected = [x for x in Path(args.samples).read_text().split() if x]
    vcf = VCF(args.vcf, samples=selected, gts012=True)
    sample_ids = list(vcf.samples)
    if args.max_samples and len(sample_ids) > args.max_samples:
        sample_ids = sample_ids[:args.max_samples]
        vcf.close()
        vcf = VCF(args.vcf, samples=sample_ids, gts012=True)

    columns = {chrom: [] for chrom in args.chrom}
    positions = {chrom: [] for chrom in args.chrom}
    biallelic = {chrom: 0 for chrom in args.chrom}
    scanned = {chrom: 0 for chrom in args.chrom}
    completed = set()
    current = None
    for variant in vcf:
        chrom = variant.CHROM
        if current is not None and chrom != current and current in requested:
            completed.add(current)
            print(f"completed {current}: kept {len(columns[current]):,}", flush=True)
        current = chrom
        if chrom not in requested:
            if completed == requested:
                break
            continue
        scanned[chrom] += 1
        if not variant.is_snp or variant.ALT is None or len(variant.ALT) != 1 \
                or len(variant.REF) != 1:
            continue
        dosage = np.asarray(variant.gt_types, dtype=np.int8)
        missing = dosage == 3
        if missing.mean() > args.max_missing:
            continue
        dosage = np.where(missing, 0, dosage).astype(np.int8)
        allele_frequency = dosage.sum() / (2.0 * len(sample_ids))
        maf = min(allele_frequency, 1.0 - allele_frequency)
        if maf < args.min_maf or maf <= 0:
            continue
        biallelic[chrom] += 1
        if args.thin > 1 and (biallelic[chrom] - 1) % args.thin:
            continue
        if args.mode == "haploid":
            row = (dosage >= 1).astype(np.int8)
        else:
            row = np.zeros(2 * len(sample_ids), dtype=np.int8)
            row[0::2] = dosage >= 1
            row[1::2] = dosage == 2
        if 0 < row.sum() < len(row):
            columns[chrom].append(row)
            positions[chrom].append(variant.POS)
        if scanned[chrom] % 500_000 == 0:
            print(f"{chrom}: scanned {scanned[chrom]:,}, kept {len(columns[chrom]):,}",
                  flush=True)
    vcf.close()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"source_vcf": args.vcf, "sample_ids": sample_ids, "mode": args.mode,
                "mu": args.mu, "thin": args.thin, "max_missing": args.max_missing,
                "min_maf": args.min_maf, "chromosomes": {}}
    for chrom in args.chrom:
        gm = np.asarray(columns[chrom], dtype=np.int8).T
        pos = np.asarray(positions[chrom], dtype=np.int64)
        if gm.size == 0:
            raise RuntimeError(f"no variants retained for {chrom}")
        key = f"{args.prefix}__{safe_name(chrom)}"
        payload = dict(gm=gm, pos=pos, chrom=chrom, n_ind=len(sample_ids), mode=args.mode,
                       mu=args.mu, sample_ids=np.asarray(sample_ids),
                       missing_policy="reference", max_missing=args.max_missing,
                       extraction_thin=args.thin)
        if args.map_sp and args.map_id:
            payload.update(map_sp=args.map_sp, map_id=args.map_id)
        path = output / f"{key}.npz"
        np.savez_compressed(path, **payload)
        manifest["chromosomes"][chrom] = {"key": key, "n_hap": int(gm.shape[0]),
                                          "n_snp": int(gm.shape[1]),
                                          "first_position": int(pos[0]),
                                          "last_position": int(pos[-1])}
        print(f"wrote {path}: {gm.shape[0]} x {gm.shape[1]:,}", flush=True)
    with open(output / f"{args.prefix}__extraction.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
