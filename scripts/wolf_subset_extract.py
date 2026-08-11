"""Extract geographically defined wolf subsets from the Plassais canid VCF.

The output schema matches ``scripts/realdata_extract.py`` so the frozen canid model can be
applied with ``scripts/realdata_infer.py <key> dogbn``. Sample identifiers are supplied in a
plain-text file, one per line, to keep subset membership explicit and auditable.
"""

from __future__ import annotations

import argparse
import os
import ssl

import certifi
import numpy as np
import pysam


VCF = "https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("id_file")
    parser.add_argument("--out-dir", default="/home/kkor/realdata/hap")
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=40_000_000)
    args = parser.parse_args()

    os.environ["CURL_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    ssl.create_default_context(cafile=certifi.where())

    requested = [line.strip() for line in open(args.id_file) if line.strip()]
    vf = pysam.VariantFile(VCF)
    available = set(vf.header.samples)
    missing = [sample for sample in requested if sample not in available]
    if missing:
        raise ValueError(f"samples absent from VCF: {missing}")

    columns, positions = [], []
    scanned = 0
    for record in vf.fetch(args.chrom, args.start, args.end):
        scanned += 1
        if record.alts is None or len(record.alts) != 1 or len(record.ref) != 1 or len(record.alts[0]) != 1:
            continue
        dosage = np.zeros(len(requested), np.int8)
        complete = True
        for index, sample in enumerate(requested):
            genotype = record.samples[sample].get("GT", (None, None))
            if len(genotype) < 2 or genotype[0] is None or genotype[1] is None:
                complete = False
                break
            dosage[index] = int(genotype[0] > 0) + int(genotype[1] > 0)
        if not complete:
            continue
        pseudo = np.zeros(2 * len(requested), np.int8)
        pseudo[0::2] = dosage >= 1
        pseudo[1::2] = dosage == 2
        if 0 < pseudo.sum() < pseudo.size:
            columns.append(pseudo)
            positions.append(record.pos)
        if scanned % 200_000 == 0:
            print(f"scanned={scanned} retained={len(columns)}", flush=True)

    matrix = np.asarray(columns, np.int8).T
    positions = np.asarray(positions, np.int64)
    os.makedirs(args.out_dir, exist_ok=True)
    output = os.path.join(args.out_dir, f"{args.key}.npz")
    np.savez(
        output,
        gm=matrix,
        pos=positions,
        chrom=args.chrom,
        n_ind=len(requested),
        mode="dosage",
        map_sp="CanFam",
        map_id="Campbell2016_CanFam3_1",
        mu=4.0e-9,
        model="dogbn",
        pop=args.key,
        sample_ids=np.asarray(requested),
    )
    print(f"{args.key}: {matrix.shape} -> {output}")


if __name__ == "__main__":
    main()
