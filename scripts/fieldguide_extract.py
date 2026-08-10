"""Extract a genotype/dosage matrix from a real multi-sample VCF for the field-guide demo.

Reads with cyvcf2 SEQUENTIALLY (no tabix index required -- sesame has no bcftools/tabix), keeps
biallelic SNPs on the requested scaffold within [start,end], optionally restricts to a sample list,
and writes hap/<key>.npz (gm, pos) in the layout scripts/fieldguide_run.py expects.

modes:
  dosage   unphased diploid -> pseudo-hap pairs preserving dosage (composite-LD model; DEFAULT,
           the common non-model case; matches scripts/realdata_extract.py dosage encoding)
  phased2  phased diploid   -> 2 haplotypes per sample
  haploid  homozygous lines -> 1 allele per sample

Run on sesame:
  PYTHONNOUSERSITE=1 ~/venvs/fastrho/bin/python scripts/fieldguide_extract.py \
      --vcf /home/kkor/realdata/<taxon>/<file>.vcf.gz --chrom <scaffold> \
      --start 1 --end 40000000 --mode dosage --samples ids.txt \
      --out /home/kkor/realdata/hap/<key>.npz
"""
import os, argparse
import numpy as np
from cyvcf2 import VCF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--chrom", required=True)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10**12)
    ap.add_argument("--mode", default="dosage", choices=["dosage", "phased2", "haploid"])
    ap.add_argument("--samples", default="", help="optional file with one sample ID per line")
    ap.add_argument("--max-missing", type=float, default=0.1, help="drop SNPs with >frac missing GT")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = None
    if args.samples:
        want = [s for s in open(args.samples).read().split() if s]
        vcf = VCF(args.vcf, samples=want, gts012=True)
    else:
        vcf = VCF(args.vcf, gts012=True)
    samples = list(vcf.samples)
    print(f"[extract] {os.path.basename(args.vcf)} scaffold={args.chrom} "
          f"[{args.start},{args.end}] mode={args.mode} samples={len(samples)}")

    cols, pos = [], []
    n_seen = n_snp = 0
    started = False
    for v in vcf:                                   # sequential scan; no index needed
        if v.CHROM != args.chrom:
            if started:
                break                               # VCF is contig-blocked: past the target scaffold
            continue
        started = True
        if v.POS < args.start or v.POS > args.end:
            continue
        n_seen += 1
        if not v.is_snp or v.ALT is None or len(v.ALT) != 1 or len(v.REF) != 1:
            continue
        # gts012: 0=hom-ref, 1=het, 2=hom-alt, 3=missing
        g = np.asarray(v.gt_types, dtype=np.int8)
        miss = g == 3
        if miss.mean() > args.max_missing:
            continue
        n_snp += 1
        d = np.where(miss, 0, g).astype(np.int8)    # ALT dosage 0/1/2 (missing -> ref)
        if args.mode == "phased2":
            gt = np.asarray(v.genotypes, dtype=object)
            row = np.zeros(2 * len(samples), np.int8)
            for k in range(len(samples)):
                a = gt[k]
                row[2 * k] = 1 if a[0] == 1 else 0
                row[2 * k + 1] = 1 if (len(a) > 1 and a[1] == 1) else 0
        elif args.mode == "haploid":
            row = (d >= 1).astype(np.int8)          # one allele per (homozygous) line
        else:                                        # dosage -> phase-invariant pseudo-haps
            row = np.zeros(2 * len(samples), np.int8)
            row[0::2] = (d >= 1).astype(np.int8)
            row[1::2] = (d == 2).astype(np.int8)
        m = int(row.sum())
        if 0 < m < len(row):                         # segregating in the subsample
            cols.append(row); pos.append(v.POS)
        if n_seen % 200000 == 0:
            print(f"  scanned {n_seen} on-scaffold records, kept {len(cols)}")

    H = np.array(cols, np.int8).T                    # (n_hap, n_snp)
    pos = np.asarray(pos, np.int64)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, gm=H, pos=pos, chrom=args.chrom, n_ind=len(samples), mode=args.mode)
    print(f"[extract] {H.shape[0]} hap x {H.shape[1]} SNP "
          f"(scanned {n_seen}, biallelic-SNP {n_snp}) -> {args.out}")


if __name__ == "__main__":
    main()
