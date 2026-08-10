"""Extract a dosage/genotype matrix from a real multi-sample VCF for the tree-of-life transect.

Generalizes scripts/fieldguide_extract.py: sequential cyvcf2 scan (no tabix needed), biallelic SNPs on
one contig within [start,end], optional sample subset / cap, and it stamps the npz with the fields
scripts/transect_infer.py needs (mu, and optionally map_sp/map_id for a stdpopsim-validated species).

modes (same encoding as fieldguide_extract / realdata_extract):
  dosage   unphased diploid -> phase-invariant pseudo-hap pair per sample (composite-LD; DEFAULT)
  phased2  phased diploid   -> 2 haplotypes per sample
  haploid  homozygous lines -> 1 allele per sample

Run on sesame (from /home/kkor/fastrho — pure cyvcf2/numpy, no fastrho package needed):
  PYTHONNOUSERSITE=1 ~/venvs/fastrho/bin/python scripts/transect_extract.py \
      --vcf <file.vcf.gz> --chrom <contig> --start 1 --end 40000000 --mode dosage \
      --mu 1.4e-8 --map-sp HomSap --map-id HapMapII_GRCh37 \
      --max-samples 80 --out /home/kkor/realdata/hap/<key>.npz
"""
import os
import argparse

import numpy as np
from cyvcf2 import VCF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--chrom", default="")
    ap.add_argument("--top-contigs", type=int, default=0, help="pool the N largest contigs (positions "
                    "offset per contig so they read as unlinked) — rescues scaffold-level assemblies")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10**12)
    ap.add_argument("--mode", default="dosage", choices=["dosage", "phased2", "haploid"])
    ap.add_argument("--samples", default="", help="optional file with one sample ID per line")
    ap.add_argument("--sample-regex", default="", help="keep only samples whose ID matches this regex "
                    "(subset ONE population/breed/locality by name); applied after --samples")
    ap.add_argument("--max-samples", type=int, default=0, help="cap #samples (take first N of the kept set)")
    ap.add_argument("--max-missing", type=float, default=0.1, help="drop SNPs with >frac missing GT")
    ap.add_argument("--min-maf", type=float, default=0.0, help="drop SNPs below this minor-allele freq")
    ap.add_argument("--max-snps", type=int, default=0, help="stop after this many kept SNPs (0=all)")
    ap.add_argument("--thin", type=int, default=1, help="keep every Kth biallelic SNP (spreads window "
                    "coverage across a dense large chromosome; 1=keep all)")
    ap.add_argument("--mu", type=float, default=1.5e-8, help="per-bp per-gen mutation rate for this species")
    ap.add_argument("--map-sp", default="", help="stdpopsim species id for validation (optional)")
    ap.add_argument("--map-id", default="", help="stdpopsim genetic-map id for validation (optional)")
    ap.add_argument("--region-tag", default="", help="free-text note on the region/population")
    ap.add_argument("--indexed", action="store_true",
                    help="use the tabix index to iterate only [chrom:start-end] (works on a remote "
                         "bgzipped+.tbi URL, avoiding a whole-genome download)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    want = None
    if args.samples:
        want = [s for s in open(args.samples).read().split() if s]
        vcf = VCF(args.vcf, samples=want, gts012=True)
    else:
        vcf = VCF(args.vcf, gts012=True)
    samples = list(vcf.samples)
    if args.sample_regex:                                # subset ONE population by name pattern
        import re
        pat = re.compile(args.sample_regex)
        keep = [s for s in samples if pat.search(s)]
        if not keep:
            raise SystemExit(f"--sample-regex '{args.sample_regex}' matched 0 of {len(samples)} samples")
        vcf.close(); vcf = VCF(args.vcf, samples=keep, gts012=True); samples = list(vcf.samples)
    if args.max_samples and len(samples) > args.max_samples:
        keep = samples[:args.max_samples]
        vcf.close()
        vcf = VCF(args.vcf, samples=keep, gts012=True)
        samples = list(vcf.samples)
    # multi-contig pooling: pick the N largest contigs and lay them end-to-end with a fixed GAP
    # between them. The gap (>> the featurizer's LD radius) keeps the composite-LD featurizer from
    # linking SNPs across a contig boundary, while CONCATENATING (offset = cumulative length, not a
    # fixed 800 Mb/contig) keeps the total span ~= genome size so the 100 kb windowing stays valid
    # for fragmented scaffold assemblies (a fixed huge offset spread SNPs over a 16 Gb span -> 0 windows).
    offset = {}; contig_set = None
    GAP = 20_000_000
    if args.top_contigs:
        # DATA pre-scan: header seqnames often DON'T match the record CHROM strings (and some VCFs
        # carry no seqlens at all), so pick the top-N contigs by ACTUAL biallelic-SNP count from a
        # streaming pass, and offset each by cumulative max-POS + GAP (concatenate, span ~= genome).
        from collections import Counter
        cnt = Counter(); maxpos = {}
        scanv = VCF(args.vcf, samples=list(vcf.samples), gts012=True)
        for v in scanv:
            if v.is_snp and v.ALT is not None and len(v.ALT) == 1 and len(v.REF) == 1:
                cnt[v.CHROM] += 1
                if v.POS > maxpos.get(v.CHROM, 0):
                    maxpos[v.CHROM] = v.POS
        scanv.close()
        top = [c for c, _ in cnt.most_common(args.top_contigs)]
        if not top:
            raise SystemExit("--top-contigs: pre-scan found no biallelic SNPs")
        contig_set = set(top)
        cum = 0
        for c in top:
            offset[c] = cum
            cum += int(maxpos.get(c, 0)) + GAP
        print(f"[extract] pooling {len(top)} contigs (data-scan, gap {GAP//10**6}Mb, span {cum//10**6}Mb): "
              f"{[f'{c}:{cnt[c]}' for c in top[:4]]}...")
    print(f"[extract] {os.path.basename(args.vcf)} contig={args.chrom or 'TOP'+str(args.top_contigs)} "
          f"mode={args.mode} samples={len(samples)} mu={args.mu}")

    cols, pos = [], []
    n_seen = n_snp = 0
    started = False
    if args.indexed and not args.top_contigs:
        end = args.end if args.end < 10**12 else ""
        iterator = vcf(f"{args.chrom}:{args.start}-{end}")   # tabix region query (local or remote URL)
    else:
        iterator = vcf                                       # sequential whole-file scan
    for v in iterator:
        if contig_set is not None:
            if v.CHROM not in contig_set:
                continue                            # multi-contig: keep only the top contigs
        elif not args.indexed and v.CHROM != args.chrom:
            if started:
                break                               # contig-blocked VCF: past the target contig
            continue
        started = True
        if contig_set is None and not args.indexed:
            if v.POS < args.start:
                continue
            if v.POS > args.end:
                break                               # positions sorted within a contig -> past region
        n_seen += 1
        if not v.is_snp or v.ALT is None or len(v.ALT) != 1 or len(v.REF) != 1:
            continue
        g = np.asarray(v.gt_types, dtype=np.int8)   # gts012: 0,1,2 dosage; 3=missing
        miss = g == 3
        if miss.mean() > args.max_missing:
            continue
        d = np.where(miss, 0, g).astype(np.int8)    # ALT dosage 0/1/2 (missing -> ref)
        if args.min_maf > 0:
            af = d.sum() / (2.0 * len(samples))
            if min(af, 1 - af) < args.min_maf:
                continue
        n_snp += 1
        if args.thin > 1 and (n_snp - 1) % args.thin != 0:   # spread coverage across the contig
            continue
        if args.mode == "phased2":
            gt = np.asarray(v.genotypes, dtype=object)
            row = np.zeros(2 * len(samples), np.int8)
            for k in range(len(samples)):
                a = gt[k]
                row[2 * k] = 1 if a[0] == 1 else 0
                row[2 * k + 1] = 1 if (len(a) > 1 and a[1] == 1) else 0
        elif args.mode == "haploid":
            row = (d >= 1).astype(np.int8)
        else:                                        # dosage -> phase-invariant pseudo-haps
            row = np.zeros(2 * len(samples), np.int8)
            row[0::2] = (d >= 1).astype(np.int8)
            row[1::2] = (d == 2).astype(np.int8)
        m = int(row.sum())
        if 0 < m < len(row):                         # segregating in the subsample
            cols.append(row); pos.append(v.POS + offset.get(v.CHROM, 0))
        if args.max_snps and len(cols) >= args.max_snps:
            print(f"  hit --max-snps={args.max_snps}, stopping"); break
        if n_seen % 200000 == 0:
            print(f"  scanned {n_seen} on-contig records, kept {len(cols)}")

    H = np.array(cols, np.int8).T                    # (n_hap, n_snp)
    pos = np.asarray(pos, np.int64)
    meta = dict(gm=H, pos=pos, chrom=(args.chrom or f"pooled{args.top_contigs}"),
                n_ind=len(samples), mode=args.mode,
                mu=args.mu, region_tag=args.region_tag)
    if args.map_sp and args.map_id:
        meta["map_sp"] = args.map_sp; meta["map_id"] = args.map_id
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **meta)
    print(f"[extract] {H.shape[0]} hap x {H.shape[1]} SNP (scanned {n_seen}, biallelic-SNP {n_snp}) "
          f"-> {args.out}"
          + (f"  [validate vs {args.map_sp}/{args.map_id}]" if args.map_sp else "  [novel — no map]"))


if __name__ == "__main__":
    main()
