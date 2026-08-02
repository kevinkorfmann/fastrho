# Quickstart

This page takes one cohort VCF to a tidy table and a browser-ready BED file.

:::{tip}
Prefer a known-answer check? {doc}`simulation` starts from an explicit `msprime.RateMap`, runs
fastrho, and evaluates the inferred map against the exact generative truth.
:::

## 1. Install

Inference is tested on Python 3.10 and 3.12 on Linux and requires an NVIDIA GPU plus a CUDA
toolchain compatible with PyTorch and Mamba-SSM. Use the locked `uv` environment:

```bash
git clone https://github.com/kevinkorfmann/fastrho.git
cd fastrho
uv sync --frozen --extra inference --extra io
source .venv/bin/activate
```

There is not yet a public PyPI package; do not install an unrelated package with a similar name.
`uv` is recommended because the project lockfile records the extension packages' build-time Torch
and NumPy needs.

You also need two files from the **same trained model release**:

- `model.ckpt` — network weights and architecture configuration
- `feat_stats.npz` — the model's feature scaling and featurizer metadata

`feat_stats.npz` is not calculated from your VCF and is not something you adapt for a new species
or cohort. It is a required companion to `model.ckpt`; download both together and leave it
unchanged. See {ref}`feat-stats-file` for its exact contents and the retraining case.

:::{tip}
Download and verify the current model in one command:

```bash
python3 scripts/fetch_model_release.py --model-id domain-randomized-v1 \
  --output-dir downloaded-models
```

The two inference paths are then
`downloaded-models/domain-randomized-v1/model.ckpt` and
`downloaded-models/domain-randomized-v1/feat_stats.npz`.
:::

## 2. Inspect the VCF before inference

This check is CPU-only and does not load Torch:

```python
import fastrho

print(fastrho.vcf_contigs("cohort.vcf.gz"))

gm, positions, meta = fastrho.read_vcf(
    "cohort.vcf.gz",
    contig="chr1",
    missing="drop-site",
    return_metadata=True,
)

print(gm.shape)   # (2 × diploid samples, retained SNPs)
print(meta)       # contig, phase status, number of dropped missing sites
```

A usable call must resolve to one contig, at least two segregating SNPs, strictly increasing
positions, and binary genotypes. Multiallelic records and indels are skipped; sites with any missing
genotype are dropped by default and are never filled in as reference.

## 3. Make the map

```python
from pathlib import Path
import fastrho

bundle = Path("downloaded-models/domain-randomized-v1")

pred = fastrho.quick_map_from_vcf(
    "cohort.vcf.gz",
    bundle / "model.ckpt",
    bundle / "feat_stats.npz",
    contig="chr1",
    mutation_rate=1.5e-8,
    Ne=10_000,             # omit only if you accept the model's point estimate
    input_mode="auto",     # detects phased vs unphased VCF genotypes
    missing="drop-site",
    device="cuda:0",
)

df = fastrho.to_dataframe(pred, chrom="chr1")
print(df[["chrom", "pos_left", "pos_right", "rho_per_bp", "r_per_bp"]].head())
print("Ne used:", df.attrs["Ne_used"])
```

`input_mode="auto"` detects separators (`|` versus `/`), but it **cannot detect whether the allele
orientation is ancestral**. If polarization is unavailable, use `input_mode="unpolarized"` and a
checkpoint whose registry metadata explicitly supports that view.

## 4. Save or plot

```python
fastrho.write_bed(pred, "chr1.intervals.bed", chrom="chr1")
fastrho.write_bed(pred, "chr1.50kb.bed", chrom="chr1", window_size=50_000)

df["cM_per_Mb"] = df["r_per_bp"] * 1e8
```

The unbinned BED contains $\rho$, $r$, and conditional $r$ limits. With `window_size`, `write_bed`
uses a span-weighted mean and writes one absolute-rate column per fixed physical window.

## 5. Sanity-check the result

Before interpreting biological differences:

1. confirm the checkpoint supports the observed phase/polarization view and plausible sample size;
2. record the checkpoint and statistics hashes, mutation rate, $N_e$, cohort filters, and window size;
3. compare maps from two disjoint sample subsets;
4. check a known chromosome-scale feature or an independent map when one exists;
5. treat split-sample agreement as repeatability, not proof of accuracy.

Next: {doc}`simulation` for a quantitative known-answer test, {doc}`python-api` for whole-genome and
in-memory workflows, or {doc}`your-data` for dataset selection and troubleshooting.
