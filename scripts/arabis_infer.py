"""Infer chromosome-wide Arabis maps from homozygous selfer WGS genotypes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import numpy as np

from fastrho.translate import load_model, predict_map_from_genotype_matrix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_selfer_vcf(path: Path, chrom: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one allele per predominantly selfing accession.

    The upstream VCF gate requires complete, homozygous calls, so this operation
    does not phase heterozygotes, duplicate diploid chromosomes, or impute calls.
    """
    rows, positions = [], []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if fields[0] != chrom or len(fields[3]) != 1 or len(fields[4]) != 1:
                continue
            gt_index = fields[8].split(":").index("GT")
            alleles = []
            valid = True
            for sample in fields[9:]:
                gt = sample.split(":")[gt_index].replace("|", "/").split("/")
                if len(gt) != 2 or gt[0] != gt[1] or gt[0] not in {"0", "1"}:
                    valid = False
                    break
                alleles.append(int(gt[0]))
            if valid and 0 < sum(alleles) < len(alleles):
                rows.append(alleles)
                positions.append(int(fields[1]) - 1)
    if len(rows) < 2:
        raise ValueError(f"too few retained SNPs in {path}:{chrom}")
    return np.asarray(rows, np.int8).T, np.asarray(positions, np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("/home/kkor/arabis_eval"))
    parser.add_argument("--maps-dir", type=Path, default=None,
                        help="output directory (default: WORKDIR/maps)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mutation-rate", type=float, default=7.0e-9)
    parser.add_argument(
        "--condition-line-multiplier", type=float, default=1.0,
        help="multiplier applied only to the checkpoint's declared n-haplotype condition",
    )
    parser.add_argument(
        "--panel", action="append", default=[], metavar="NAME=VCF",
        help="infer only an explicitly named VCF (repeatable); VCF may be absolute",
    )
    args = parser.parse_args()

    model, config, stats = load_model(str(args.checkpoint), str(args.stats), args.device)
    checkpoint_sha256 = sha256(args.checkpoint)
    stats_sha256 = sha256(args.stats)
    outdir = args.maps_dir or args.workdir / "maps"
    outdir.mkdir(parents=True, exist_ok=True)
    panels = ({item.split("=", 1)[0]: item.split("=", 1)[1] for item in args.panel}
              if args.panel else {
                  "nemorensis": "nemorensis.selfer.vcf.gz",
                  "sagittata": "sagittata.selfer.vcf.gz",
                  "nemorensis_no_cross_parent": "nemorensis.no_cross_parent.selfer.vcf.gz",
                  "sagittata_no_cross_parent": "sagittata.no_cross_parent.selfer.vcf.gz",
              })
    for panel, filename in panels.items():
        species = panel.split("_", 1)[0]
        vcf = Path(filename)
        if not vcf.is_absolute():
            vcf = args.workdir / "vcf" / vcf
        for number in range(1, 9):
            chrom = f"chr{number}"
            gm, positions = read_selfer_vcf(vcf, chrom)
            pred = predict_map_from_genotype_matrix(
                gm,
                positions,
                model,
                config,
                stats,
                mutation_rate=args.mutation_rate,
                device=args.device,
                input_mode="phased",
                n_hap_condition=int(round(gm.shape[0] * args.condition_line_multiplier)),
            )
            np.savez_compressed(
                outdir / f"{panel}.{chrom}.npz",
                **{k: v for k, v in pred.items() if isinstance(v, (np.ndarray, int, float))},
                species=species,
                panel=panel,
                chrom=chrom,
                n_accessions=gm.shape[0],
                n_snps=gm.shape[1],
                mutation_rate=args.mutation_rate,
                condition_line_multiplier=args.condition_line_multiplier,
                input_definition="one_homozygous_allele_per_accession",
                checkpoint_sha256=checkpoint_sha256,
                stats_sha256=stats_sha256,
            )
            print(
                f"{panel} {chrom}: {gm.shape[0]} accessions, {gm.shape[1]} SNPs, "
                f"Ne={float(pred['Ne_estimated']):.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
