"""Autonomously probe EBI EVA studies for clean single-species genotype VCFs and emit the transect
ingest config + metadata. For each (key, accession, ...) it lists the EVA FTP dir, picks the most
likely multi-sample callset VCF, streams its header to confirm per-sample GT + read the sample count
and the largest contig, and (if it passes basic checks) writes a master-config line + a meta entry.

The ingest/infer/QC/gate pipeline downstream enforces the real quality bar (n_hap>=40, n_snp>=8k,
blind-reproducibility>=0.5, outbred-only for novel). This just finds and characterises the file.

Run on sesame: python scripts/eva_probe.py <species.tsv> <out_cfg> <out_meta.json>
  species.tsv cols (tab): key  accession  common  latin  clade  mu  regime  order_idx
"""
import sys
import re
import json
import subprocess

EVA = "https://ftp.ebi.ac.uk/pub/databases/eva"
# heuristics to pick the multi-sample callset among a study's files
GOOD = re.compile(r"(merge|final|snp|all|population|joint|filter|cohort|autosom|biallelic|geno|pass|"
                  r"variant|combined|imputed|phased|wgs)", re.I)
BAD = re.compile(r"(sites?_only|sites\.|\.sites|indel|structural|README|md5|tbi|csi|\.tab|chrX|chrMT|"
                 r"array|axiom|chip|snp50|50k|60k|600k|670k|test|example)", re.I)


def sh(cmd, t=60):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t).stdout
    except Exception:
        return ""


def list_vcfs(acc):
    html = sh(f'curl -s "{EVA}/{acc}/"', 40)
    fs = re.findall(r'href="([^"]+\.vcf\.gz)"', html)
    fs = [f for f in fs if not BAD.search(f)]
    # prefer accessioned=False (raw sample names), then GOOD-matching, then largest-name-ish
    fs.sort(key=lambda f: (("accessioned" in f), (not bool(GOOD.search(f))), len(f)))
    return fs


def header(url):
    """Stream the gzip header; return (nsamp, first_chrom, has_gt) or (0,None,False)."""
    out = sh(f'curl -s "{url}" 2>/dev/null | zcat 2>/dev/null | '
             f'awk \'/^#CHROM/{{print "NS",NF-9}} /^#CHROM/{{next}} !/^#/{{print "REC",$1,$9;exit}}\'', 90)
    nsamp = 0; chrom = None; gt = False
    for line in out.splitlines():
        if line.startswith("NS"):
            nsamp = int(line.split()[1])
        elif line.startswith("REC"):
            p = line.split()
            chrom = p[1] if len(p) > 1 else None
            gt = len(p) > 2 and p[2].startswith("GT")
    return nsamp, chrom, gt


def main():
    tsv, out_cfg, out_meta = sys.argv[1], sys.argv[2], sys.argv[3]
    cfg_lines, meta = [], {}
    for ln in open(tsv):
        ln = ln.rstrip("\n")
        if not ln or ln.startswith("#"):
            continue
        key, acc, common, latin, clade, mu, regime, oidx = (ln.split("|") + [""] * 8)[:8]
        vcfs = list_vcfs(acc)
        if not vcfs:
            print(f"[{key}] no vcf in {acc}"); continue
        chosen = None
        for f in vcfs[:4]:
            url = f"{EVA}/{acc}/{f}"
            ns, chrom, gt = header(url)
            if ns >= 20 and gt and chrom:
                chosen = (f, ns, chrom); break
            print(f"[{key}] {f}: ns={ns} chrom={chrom} gt={gt} (skip)")
        if not chosen:
            print(f"[{key}] no usable VCF"); continue
        f, ns, chrom = chosen
        url = f"{EVA}/{acc}/{f}"
        # mode: phased if 'phased' in filename else dosage
        mode = "phased2" if "phased" in f.lower() else "dosage"
        # thin so ~large chromosomes stay tractable; cap samples at 120
        cfg_lines.append(f"{key}|{url}|{chrom}|1|120000000|{mode}|{mu}|8|120|-|-|-|0.4|")
        meta[key] = dict(latin=latin, common=common, clade=clade,
                         order_idx=int(oidx) if oidx else 50, source=f"EVA {acc}",
                         n_dip=min(ns, 120), regime=regime or "outbred",
                         specialist_note=None, specialist_r=None)
        print(f"[{key}] OK {f} ns={ns} chrom={chrom} mode={mode}")
    open(out_cfg, "w").write("\n".join(cfg_lines) + "\n")
    json.dump(meta, open(out_meta, "w"), indent=1)
    print(f"\nwrote {len(cfg_lines)} config lines -> {out_cfg}; {len(meta)} meta -> {out_meta}")


if __name__ == "__main__":
    main()
