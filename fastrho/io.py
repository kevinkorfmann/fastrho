"""Lightweight, torch-free I/O for fastrho: VCF reading and tidy map output.

Kept separate from :mod:`fastrho.translate` (which imports torch) so that **reading a VCF**
or turning a prediction into a ``pandas.DataFrame`` works on any CPU machine without the GPU
inference stack (torch / mamba-ssm) installed. :mod:`fastrho.translate` calls into here.
"""

from __future__ import annotations

import os

import numpy as np

# Per-interval array keys produced by ``predict_from_tokens`` (everything that is one value
# per SNP interval, in BED order). Scalars such as Ne live outside this tuple.
INTERVAL_KEYS = ("pos_left", "pos_right", "rho_per_bp", "r_per_bp",
                 "log_rho", "sigma_log_rho", "rho_ci_lo", "rho_ci_hi",
                 "r_ci_lo", "r_ci_hi")


# ---------------------------------------------------------------------------
# VCF -> genotype matrix
# ---------------------------------------------------------------------------

def _read_vcf_plain(vcf_path, contig=None, missing="drop-site"):
    """Dependency-free phased-VCF reader: GT field -> (gm[n_hap, n_sites], positions).

    Parses the GT column directly so it works on tskit/msprime **numeric 0/1 alleles**, which
    cyvcf2's ``is_snp`` filter rejects. Transparently handles gzip. Each diploid ``a|b`` (or
    ``a/b``) column expands to two haplotype rows; biallelic single-character REF/ALT only.
    """
    import gzip
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    if missing not in {"drop-site", "error"}:
        raise ValueError("missing must be 'drop-site' or 'error'")
    haps, pos = [], []
    seen_contigs = set()
    all_phased = True
    dropped_missing = 0
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if contig and f[0] != contig:
                continue
            seen_contigs.add(f[0])
            if len(f[3]) != 1 or len(f[4]) != 1:     # biallelic single-char REF/ALT (accepts 0/1)
                continue
            fmt = f[8].split(":")
            if "GT" not in fmt:
                continue
            gt_index = fmt.index("GT")
            row = []
            site_phased = True
            site_missing = False
            for g in f[9:]:
                fields = g.split(":")
                if gt_index >= len(fields):
                    site_missing = True
                    break
                gt = fields[gt_index]
                delimiter = "|" if "|" in gt else "/"
                site_phased &= delimiter == "|"
                alleles = gt.split(delimiter)
                if len(alleles) != 2 or "." in alleles:
                    site_missing = True
                    break
                values = [int(a) for a in alleles]
                if any(a not in (0, 1) for a in values):
                    raise ValueError(f"non-biallelic genotype {gt!r} at {f[0]}:{f[1]}")
                row.extend(values)
            if site_missing:
                if missing == "error":
                    raise ValueError(f"missing genotype at {f[0]}:{f[1]}")
                dropped_missing += 1
                continue
            all_phased &= site_phased
            haps.append(row)
            pos.append(int(f[1]) - 1)  # VCF 1-based POS -> internal 0-based coordinate
    if contig is None and len(seen_contigs) > 1:
        raise ValueError(
            f"VCF contains multiple contigs {sorted(seen_contigs)}; pass contig explicitly"
        )
    if not haps:
        raise ValueError(f"no biallelic SNP records found in {vcf_path}"
                         + (f" for contig {contig!r}" if contig else ""))
    gm = np.asarray(haps, dtype=np.int8).T           # (n_hap, n_sites)
    meta = {"contig": contig or next(iter(seen_contigs)), "phased": all_phased,
            "dropped_missing_sites": dropped_missing}
    return gm, np.asarray(pos, dtype=np.float64), meta


def vcf_contigs(vcf_path):
    """List contig names declared in a VCF header (the ``##contig=<ID=...>`` lines).

    Handy for looping a genome-wide prediction over chromosomes without hard-coding names.
    Transparently handles gzip. Returns ``[]`` if the header declares no contigs (some VCFs
    don't) — fall back to an explicit list in that case.
    """
    import gzip
    import re
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    out = []
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("##contig"):
                m = re.search(r"ID=([^,>]+)", line)
                if m:
                    out.append(m.group(1))
            elif not line.startswith("#"):
                break                                # past the header
    return out


def read_vcf(vcf_path, contig=None, *, missing="drop-site", return_metadata=False):
    """Phased VCF -> ``(gm[n_hap, n_sites], positions[n_sites])``.

    Every VCF sample column is read; this function does not select, cap, or randomly subsample
    individuals. Each diploid sample expands to two adjacent allele rows, so callers must subset
    large VCFs to a deliberate cohort before reading when required by the checkpoint's training
    domain.

    Works on both real ACGT VCFs and **tskit/msprime numeric-allele (0/1) VCFs** — the kind
    ``tskit.TreeSequence.write_vcf`` emits. cyvcf2 is used when installed (fast, indexed),
    otherwise a dependency-free reader is used so the common case needs no extra packages.
    Pass ``contig`` to restrict to one chromosome. Coordinates are converted from
    VCF's 1-based POS convention to 0-based positions. Sites containing missing
    genotypes are dropped by default; ``missing='error'`` fails instead. Missing
    alleles are never imputed as reference. Set ``return_metadata=True`` to also
    receive the resolved contig, phase status, and dropped-site count.
    """
    if missing not in {"drop-site", "error"}:
        raise ValueError("missing must be 'drop-site' or 'error'")
    try:
        from cyvcf2 import VCF
    except ImportError:
        gm, pos, meta = _read_vcf_plain(vcf_path, contig=contig, missing=missing)
        return (gm, pos, meta) if return_metadata else (gm, pos)

    reader = VCF(vcf_path)
    has_index = any(os.path.exists(str(vcf_path) + suffix) for suffix in (".tbi", ".csi"))
    iterator = reader(contig) if contig and has_index else reader
    gts, pos = [], []
    seen_contigs = set()
    all_phased = True
    dropped_missing = 0
    for v in iterator:
        if contig and v.CHROM != contig:
            continue
        seen_contigs.add(v.CHROM)
        # biallelic single-char REF/ALT; accepts numeric 0/1 (cyvcf2 .is_snp would reject)
        if len(v.REF) != 1 or len(v.ALT) != 1 or len(v.ALT[0]) != 1:
            continue
        genotypes = v.genotypes                       # [allele1, allele2, phased]
        g = np.asarray([x[:2] for x in genotypes], dtype=np.int16)
        if np.any(g < 0):
            if missing == "error":
                raise ValueError(f"missing genotype at {v.CHROM}:{v.POS}")
            dropped_missing += 1
            continue
        if np.any(g > 1):
            raise ValueError(f"non-biallelic genotype at {v.CHROM}:{v.POS}")
        all_phased &= all(bool(x[2]) for x in genotypes)
        gts.append(g.reshape(-1))
        pos.append(v.POS - 1)
    if contig is None and len(seen_contigs) > 1:
        raise ValueError(
            f"VCF contains multiple contigs {sorted(seen_contigs)}; pass contig explicitly"
        )
    if not gts:
        raise ValueError(f"no biallelic SNP records found in {vcf_path}"
                         + (f" for contig {contig!r}" if contig else ""))
    gm = np.asarray(gts, dtype=np.int8).T            # (n_hap, n_sites)
    meta = {"contig": contig or next(iter(seen_contigs)), "phased": all_phased,
            "dropped_missing_sites": dropped_missing}
    result = (gm, np.asarray(pos, dtype=np.float64), meta)
    return result if return_metadata else result[:2]


# ---------------------------------------------------------------------------
# Prediction dict -> tidy table
# ---------------------------------------------------------------------------

def to_dataframe(pred: dict, chrom: str | None = None):
    """Per-interval prediction dict -> tidy ``pandas.DataFrame`` (one row per interval).

    Columns are the per-interval arrays from ``predict_from_tokens`` (interval edges,
    ``rho``/``r`` and their 95% CIs). The scalar ``Ne_used``/``Ne_estimated`` are attached on
    ``df.attrs``. Pass ``chrom`` to prepend a ``chrom`` column for BED-like downstream use.
    """
    import pandas as pd
    df = pd.DataFrame({k: pred[k] for k in INTERVAL_KEYS if k in pred})
    if chrom is not None:
        df.insert(0, "chrom", chrom)
    df.attrs["Ne_used"] = pred.get("Ne_used")
    df.attrs["Ne_estimated"] = pred.get("Ne_estimated")
    df.attrs["coordinate_system"] = pred.get("coordinate_system", "0-based-half-open")
    df.attrs["r_interval_is_conditional_on_Ne"] = pred.get("r_interval_is_conditional_on_Ne")
    return df
