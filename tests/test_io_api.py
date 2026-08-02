"""CPU-only tests for the torch-free Python API: VCF reading, tidy output, lazy imports.

These cover exactly the "get a map from a VCF" ergonomics documented in docs/python-api.md
and the example notebooks, and run without torch/CUDA (unlike test_phase4_translate.py).
"""

import gzip
import subprocess
import sys

import numpy as np
import pytest


def _write_numeric_vcf(path, n_indiv=4, positions=(10, 25, 40, 55), chrom="chr1", gzipped=False):
    """A minimal tskit/msprime-style phased VCF with numeric 0/1 alleles."""
    samples = [f"tsk_{i}" for i in range(n_indiv)]
    rng = np.random.default_rng(0)
    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={chrom}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    for p in positions:
        gts = "\t".join(f"{rng.integers(0, 2)}|{rng.integers(0, 2)}" for _ in samples)
        lines.append(f"{chrom}\t{p}\t.\t0\t1\t.\tPASS\t.\tGT\t{gts}")
    text = "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return samples, list(positions)


def test_read_vcf_numeric_alleles(tmp_path):
    from fastrho.io import read_vcf
    p = tmp_path / "region.vcf"
    samples, positions = _write_numeric_vcf(p, n_indiv=4, positions=(10, 25, 40, 55))
    gm, pos = read_vcf(str(p), contig="chr1")
    assert gm.shape == (2 * len(samples), len(positions))   # (n_hap, n_sites)
    assert gm.dtype == np.int8
    assert set(np.unique(gm)).issubset({0, 1})
    assert pos.tolist() == [float(x - 1) for x in positions]


def test_read_vcf_gzip_and_contig_filter(tmp_path):
    from fastrho.io import read_vcf
    p = tmp_path / "region.vcf.gz"
    _write_numeric_vcf(p, positions=(5, 15, 30), chrom="2L", gzipped=True)
    gm, pos = read_vcf(str(p), contig="2L")
    assert gm.shape[1] == 3
    with pytest.raises(ValueError):              # no records for a missing contig
        read_vcf(str(p), contig="chrZ")


def test_read_vcf_skips_indels_and_multiallelic(tmp_path):
    from fastrho.io import read_vcf
    p = tmp_path / "mixed.vcf"
    lines = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1",
        "chr1\t10\t.\t0\t1\t.\tPASS\t.\tGT\t0|1\t1|0",   # kept (biallelic SNP)
        "chr1\t20\t.\tAT\tA\t.\tPASS\t.\tGT\t0|1\t1|1",  # skipped (indel: len(REF)!=1)
        "chr1\t30\t.\t0\t1,2\t.\tPASS\t.\tGT\t0|1\t2|1",  # skipped (multiallelic: len(ALT)!=1)
        "chr1\t40\t.\t0\t1\t.\tPASS\t.\tGT\t1|1\t0|0",   # kept
    ]
    p.write_text("\n".join(lines) + "\n")
    gm, pos = read_vcf(str(p))
    assert pos.tolist() == [9.0, 39.0]
    assert gm.shape == (4, 2)


def test_vcf_contigs(tmp_path):
    from fastrho.io import vcf_contigs
    p = tmp_path / "genome.vcf"
    p.write_text("\n".join([
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1,length=1000000>",
        "##contig=<ID=chr2,length=2000000>",
        "##contig=<ID=chrX,length=1500000>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',   # a meta line after contigs
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0",
        "chr1\t10\t.\t0\t1\t.\tPASS\t.\tGT\t0|1",
    ]) + "\n")
    assert vcf_contigs(str(p)) == ["chr1", "chr2", "chrX"]

    pg = tmp_path / "genome.vcf.gz"
    with gzip.open(pg, "wt") as fh:
        fh.write(p.read_text())
    assert vcf_contigs(str(pg)) == ["chr1", "chr2", "chrX"]

    nohdr = tmp_path / "nohdr.vcf"
    nohdr.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                     "chr1\t10\t.\t0\t1\t.\tPASS\t.\n")
    assert vcf_contigs(str(nohdr)) == []          # no ##contig headers → empty


def test_read_vcf_drops_missing_without_reference_imputation(tmp_path):
    from fastrho.io import read_vcf
    p = tmp_path / "missing.vcf"
    p.write_text("\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1",
        "chr1\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|0",
        "chr1\t20\t.\tA\tG\t.\tPASS\t.\tGT\t0|.\t1|0",
        "chr1\t30\t.\tA\tG\t.\tPASS\t.\tGT\t1|1\t0|0",
    ]) + "\n")
    gm, pos, meta = read_vcf(p, return_metadata=True)
    assert pos.tolist() == [9.0, 29.0]
    assert gm.shape == (4, 2)
    assert meta["dropped_missing_sites"] == 1
    with pytest.raises(ValueError, match="missing genotype"):
        read_vcf(p, missing="error")


def test_read_vcf_requires_one_contig_and_reports_phase(tmp_path):
    from fastrho.io import read_vcf
    p = tmp_path / "two.vcf"
    p.write_text("\n".join([
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0",
        "chr1\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0/1",
        "chr2\t20\t.\tA\tG\t.\tPASS\t.\tGT\t0/1",
    ]) + "\n")
    with pytest.raises(ValueError, match="multiple contigs"):
        read_vcf(p)
    _, _, meta = read_vcf(p, contig="chr1", return_metadata=True)
    assert meta["contig"] == "chr1"
    assert meta["phased"] is False


def test_to_dataframe_columns_and_attrs():
    pytest.importorskip("pandas")
    from fastrho.io import INTERVAL_KEYS, to_dataframe
    n = 5
    pred = {k: np.linspace(0, 1, n) for k in INTERVAL_KEYS}
    pred["Ne_used"], pred["Ne_estimated"] = 1.0e4, 1.2e4
    df = to_dataframe(pred, chrom="chr1")
    assert len(df) == n
    assert list(df.columns)[0] == "chrom"
    assert set(INTERVAL_KEYS).issubset(df.columns)
    assert df.attrs["Ne_used"] == 1.0e4 and df.attrs["Ne_estimated"] == 1.2e4


def test_import_fastrho_is_torch_free():
    """A bare ``import fastrho`` (and reaching torch-free helpers) must not import torch.

    The docs build and CPU tooling rely on this; run in a subprocess so collection order of
    the GPU test modules can't pollute sys.modules.
    """
    code = (
        "import sys, fastrho;"
        "assert hasattr(fastrho, '__version__');"
        "assert 'torch' not in sys.modules, 'import fastrho pulled torch';"
        "assert 'fastrho.translate' not in sys.modules, 'import fastrho pulled translate';"
        "fastrho.read_vcf; fastrho.to_dataframe;"          # torch-free helpers resolve...
        "assert 'torch' not in sys.modules, 'torch-free helpers pulled torch'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
