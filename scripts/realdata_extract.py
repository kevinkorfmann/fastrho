"""Extract a haplotype/dosage matrix from REAL genotype data for fastrho validation.

Streams a chromosome region from a (possibly remote) VCF via pysam, selects a population,
and writes haps/<key>.npz with the matrix + positions, in the assembly that matches the
published recombination map we will compare against.

modes: phased2 (diploid phased -> 2 haps/sample); dosage (unphased diploid -> pseudo-hap
pairs preserving dosage, for the composite-LD model); haploid (homozygous lines/accessions
-> 1 hap each).
"""
import os, sys
import certifi
os.environ["CURL_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()
import numpy as np
import pysam
import urllib.request, ssl

OUT = "/home/kkor/realdata/hap"
H1KG = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
PANEL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"

CFG = {
    "human": dict(path=H1KG.format(c="2"), chrom="2", start=20_000_000, end=36_000_000,
                  mode="phased2", pop="CEU", map_sp="HomSap", map_id="HapMapII_GRCh37",
                  mu=1.29e-8, model="base"),
    "dog":   dict(path="https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz",
                  chrom="chr1", start=1, end=40_000_000, mode="dosage",
                  id_file="/home/kkor/realdata/dog_china_ids.txt",
                  map_sp="CanFam", map_id="Campbell2016_CanFam3_1", mu=4.0e-9, model="gt"),
    "dmel":  dict(path="/home/kkor/realdata/dmel/DGRP2.dm6.SNPs.vcf.gz", chrom="2L",
                  start=1, end=23_500_000, mode="haploid", map_sp="DroMel",
                  map_id="ComeronCrossover_dm6", mu=5.5e-9, model="model2"),
    "athal": dict(path="/home/kkor/realdata/athal/1001g.vcf.gz", chrom="1", start=1,
                  end=20_000_000, mode="haploid", id_file="/home/kkor/realdata/athal_swe_ids.txt",
                  map_sp="AraTha", map_id="SalomeAveraged_TAIR10", mu=7.0e-9, model="base"),
}


def select_samples(cfg, vf):
    alls = list(vf.header.samples)
    if cfg.get("pop"):
        ctx = ssl.create_default_context(cafile=certifi.where())
        rows = urllib.request.urlopen(PANEL, context=ctx).read().decode().splitlines()[1:]
        want = {r.split()[0] for r in rows if len(r.split()) > 1 and r.split()[1] == cfg["pop"]}
        return [s for s in alls if s in want]
    if cfg.get("prefixes"):
        import re
        pat = re.compile("^(%s)[0-9]" % "|".join(cfg["prefixes"]))
        return [s for s in alls if pat.match(s)]
    if cfg.get("id_file"):
        want = set(open(cfg["id_file"]).read().split())
        return [s for s in alls if s in want]
    if cfg.get("breed"):
        b = cfg["breed"]
        sel = [s for s in alls if s.upper().startswith(b) and not s[len(b):len(b)+1].isalpha()]
        return sel
    if cfg.get("n_sub"):
        rng = np.random.default_rng(2026)
        return [alls[i] for i in np.sort(rng.choice(len(alls), min(cfg["n_sub"], len(alls)), replace=False))]
    return alls


def main():
    key = sys.argv[1]
    cfg = CFG[key]
    vf = pysam.VariantFile(cfg["path"])
    samples = select_samples(cfg, vf)
    print(f"{key}: {len(samples)} samples ({cfg['mode']})")
    cols, pos = [], []
    mode = cfg["mode"]
    n = 0
    for rec in vf.fetch(cfg["chrom"], cfg["start"], cfg["end"]):
        n += 1
        if rec.alts is None or len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue
        gts = rec.samples
        if mode == "phased2":
            row = np.zeros(2 * len(samples), np.int8)
            for k, s in enumerate(samples):
                a = gts[s].get("GT", (None, None))
                row[2 * k] = 1 if a[0] else 0
                row[2 * k + 1] = 1 if (len(a) > 1 and a[1]) else 0
        elif mode == "dosage":
            d = np.zeros(len(samples), np.int8)
            for k, s in enumerate(samples):
                a = gts[s].get("GT", (None, None))
                d[k] = (1 if a[0] else 0) + (1 if (len(a) > 1 and a[1]) else 0)
            # pseudo-haps preserving dosage (phase-invariant for the composite-LD model)
            row = np.zeros(2 * len(samples), np.int8)
            row[0::2] = (d >= 1).astype(np.int8)
            row[1::2] = (d == 2).astype(np.int8)
        else:  # haploid: homozygous line/accession -> one allele
            row = np.zeros(len(samples), np.int8)
            for k, s in enumerate(samples):
                a = gts[s].get("GT", (None,))
                row[k] = 1 if a[0] else 0
        m = row.sum()
        if 0 < m < len(row):                       # segregating in the subsample
            cols.append(row); pos.append(rec.pos)
        if n % 200000 == 0:
            print(f"  scanned {n} records, kept {len(cols)}")
    H = np.array(cols, np.int8).T                  # (n_hap, n_snp)
    pos = np.asarray(pos, np.int64)
    os.makedirs(OUT, exist_ok=True)
    fn = f"{OUT}/{key}.npz"
    np.savez(fn, gm=H, pos=pos, chrom=cfg["chrom"], n_ind=len(samples), mode=mode,
             map_sp=cfg["map_sp"], map_id=cfg["map_id"], mu=cfg["mu"], model=cfg["model"], pop=key)
    print(f"{key}: {H.shape[0]} hap x {H.shape[1]} SNPs ({cfg['chrom']}:{cfg['start']}-{cfg['end']}) -> {fn}")


if __name__ == "__main__":
    main()
