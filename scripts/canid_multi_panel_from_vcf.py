"""Extract several canid panels in one indexed VCF pass.

Each ``--panel`` argument supplies a key, an ID file, and an output archive.
Variant filtering and missing-data handling are identical for every panel, but
polymorphism is assessed within each panel.  This permits matched dog--wolf
analyses without downloading chromosome 1 once per panel.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np


def read_ids(path: str) -> list[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf")
    parser.add_argument("--panel", nargs=3, action="append", required=True,
                        metavar=("KEY", "ID_FILE", "OUTPUT"))
    parser.add_argument("--region", default="chr1")
    parser.add_argument("--missing-policy", choices=("complete", "reference"),
                        default="reference")
    args = parser.parse_args()

    panels = []
    requested = set()
    for key, id_file, output in args.panel:
        ids = read_ids(id_file)
        if not ids:
            raise ValueError(f"empty panel: {key}")
        panels.append({"key": key, "ids": ids, "output": output,
                       "columns": [], "positions": []})
        requested.update(ids)

    header_ids = subprocess.check_output(
        ["bcftools", "query", "-l", args.vcf], text=True).split()
    absent = requested.difference(header_ids)
    if absent:
        raise ValueError(f"samples absent from VCF: {sorted(absent)}")
    union_ids = [sample for sample in header_ids if sample in requested]
    union_lookup = {sample: index for index, sample in enumerate(union_ids)}
    for panel in panels:
        panel["indices"] = np.asarray(
            [union_lookup[sample] for sample in panel["ids"]], dtype=int)

    with tempfile.NamedTemporaryFile("w", delete=False) as sample_file:
        sample_file.write("\n".join(union_ids) + "\n")
        sample_path = sample_file.name
    command = [
        "bcftools", "query", "-S", sample_path, "-r", args.region,
        "-i", 'TYPE="snp" && N_ALT=1', "-f", r"%POS[\t%GT]\n", args.vcf,
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        for line_number, line in enumerate(process.stdout, 1):
            fields = line.rstrip().split("\t")
            if len(fields) != len(union_ids) + 1:
                raise RuntimeError(
                    f"expected {len(union_ids)} genotypes, found {len(fields) - 1}"
                )
            genotypes = fields[1:]
            position = int(fields[0])
            for panel in panels:
                selected = [genotypes[index] for index in panel["indices"]]
                if (args.missing_policy == "complete"
                        and any("." in genotype for genotype in selected)):
                    continue
                dosage = np.asarray([
                    sum(int(allele) for allele in genotype.replace("|", "/").split("/")
                        if allele != ".")
                    for genotype in selected
                ], dtype=np.int8)
                pseudo = np.zeros(2 * len(selected), np.int8)
                pseudo[0::2] = dosage >= 1
                pseudo[1::2] = dosage == 2
                if 0 < pseudo.sum() < pseudo.size:
                    panel["positions"].append(position)
                    panel["columns"].append(pseudo)
            if line_number % 200_000 == 0:
                counts = " ".join(
                    f"{panel['key']}={len(panel['columns'])}" for panel in panels)
                print(f"scanned={line_number} {counts}", flush=True)
        if process.wait() != 0:
            raise RuntimeError("bcftools query failed")
    finally:
        os.unlink(sample_path)

    chrom = args.region.split(":", 1)[0]
    for panel in panels:
        matrix = np.asarray(panel["columns"], np.int8).T
        positions = np.asarray(panel["positions"], np.int64)
        os.makedirs(os.path.dirname(panel["output"]) or ".", exist_ok=True)
        np.savez(
            panel["output"], gm=matrix, pos=positions, chrom=chrom,
            n_ind=len(panel["ids"]), mode="dosage", map_sp="CanFam",
            map_id="Campbell2016_CanFam3_1", mu=4.0e-9, model="dogbn",
            pop=panel["key"], sample_ids=np.asarray(panel["ids"]),
            missing_policy=args.missing_policy,
        )
        print(f"{panel['key']}: {matrix.shape} -> {panel['output']}")


if __name__ == "__main__":
    main()
