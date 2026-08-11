"""Extract full-chromosome haploid matrices for A. thaliana chr1-5 (Extended Data selfer figure).

Mirrors scripts/realdata_extract.py (haploid mode: homozygous accession -> one allele) but loops
over all five chromosomes and takes the FULL chromosome span so the pericentromere is included on
each. Same 156 Swedish accessions, same mu, same Salome/TAIR10 map as the paper's chr1 case.

Run in the agam venv (has pysam):
    /home/kkor/venvs/agam/bin/python athal_ed_extract.py
"""
import os
import numpy as np
import pysam

VCF = "/home/kkor/realdata/athal/1001g.vcf.gz"
IDS = "/home/kkor/realdata/athal_swe_ids.txt"
OUT = "/home/kkor/realdata/hap"
MU = 7.0e-9
MAP_SP = "AraTha"
MAP_ID = "SalomeAveraged_TAIR10"
MODEL = "self2"

# TAIR10 chromosome lengths (stdpopsim AraTha genome)
CHROM_LEN = {"1": 30427671, "2": 19698289, "3": 23459830, "4": 18585056, "5": 26975502}


def main():
    want = set(open(IDS).read().split())
    vf = pysam.VariantFile(VCF)
    samples = [s for s in vf.header.samples if s in want]
    print(f"selected {len(samples)} accessions (haploid)", flush=True)
    os.makedirs(OUT, exist_ok=True)

    for chrom, length in CHROM_LEN.items():
        cols, pos, n = [], [], 0
        for rec in vf.fetch(chrom, 0, length):
            n += 1
            if rec.alts is None or len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue
            gts = rec.samples
            row = np.zeros(len(samples), np.int8)
            for k, s in enumerate(samples):
                a = gts[s].get("GT", (None,))
                row[k] = 1 if a[0] else 0
            m = row.sum()
            if 0 < m < len(row):
                cols.append(row); pos.append(rec.pos)
            if n % 400000 == 0:
                print(f"  chr{chrom}: scanned {n}, kept {len(cols)}", flush=True)
        H = np.array(cols, np.int8).T
        p = np.asarray(pos, np.int64)
        fn = f"{OUT}/athal_c{chrom}.npz"
        np.savez(fn, gm=H, pos=p, chrom=chrom, n_ind=len(samples), mode="haploid",
                 map_sp=MAP_SP, map_id=MAP_ID, mu=MU, model=MODEL, pop=f"athal_c{chrom}")
        print(f"athal_c{chrom}: {H.shape[0]} hap x {H.shape[1]} SNPs ({chrom}:1-{length}) -> {fn}", flush=True)


if __name__ == "__main__":
    main()
