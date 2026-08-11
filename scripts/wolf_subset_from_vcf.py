"""Create a fastrho pseudo-haplotype archive from an indexed canid VCF."""

from __future__ import annotations

import argparse
import subprocess

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("vcf")
    parser.add_argument("id_file")
    parser.add_argument("output")
    parser.add_argument(
        "--region", default="chr1",
        help="Indexed VCF region to query (default: chr1)",
    )
    parser.add_argument(
        "--missing-policy", choices=("complete", "reference"), default="complete",
        help=("Drop sites with any missing genotype (complete), or encode missing "
              "alleles as reference to reproduce the original canid analysis (reference)"),
    )
    args = parser.parse_args()

    requested = [line.strip() for line in open(args.id_file) if line.strip()]
    command = [
        "bcftools", "query", "-S", args.id_file, "-r", args.region,
        "-i", 'TYPE="snp" && N_ALT=1',
        "-f", r"%POS[\t%GT]\n", args.vcf,
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    columns, positions = [], []
    for line_number, line in enumerate(process.stdout, 1):
        fields = line.rstrip().split("\t")
        if len(fields) != len(requested) + 1:
            raise RuntimeError(
                f"expected {len(requested)} genotypes, found {len(fields) - 1}"
            )
        dosage = np.zeros(len(requested), np.int8)
        complete = True
        for index, genotype in enumerate(fields[1:]):
            if "." in genotype:
                if args.missing_policy == "complete":
                    complete = False
                    break
            alleles = genotype.replace("|", "/").split("/")
            dosage[index] = sum(int(allele) for allele in alleles if allele != ".")
        if not complete:
            continue
        pseudo = np.zeros(2 * len(requested), np.int8)
        pseudo[0::2] = dosage >= 1
        pseudo[1::2] = dosage == 2
        if 0 < pseudo.sum() < pseudo.size:
            positions.append(int(fields[0]))
            columns.append(pseudo)
        if line_number % 200_000 == 0:
            print(f"scanned={line_number} retained={len(columns)}", flush=True)
    if process.wait() != 0:
        raise RuntimeError("bcftools query failed")

    matrix = np.asarray(columns, np.int8).T
    positions = np.asarray(positions, np.int64)
    np.savez(
        args.output,
        gm=matrix,
        pos=positions,
        chrom=args.region.split(":", 1)[0],
        n_ind=len(requested),
        mode="dosage",
        map_sp="CanFam",
        map_id="Campbell2016_CanFam3_1",
        mu=4.0e-9,
        model="dogbn",
        pop=args.key,
        sample_ids=np.asarray(requested),
        missing_policy=args.missing_policy,
    )
    print(f"{args.key}: {matrix.shape} -> {args.output}")


if __name__ == "__main__":
    main()
