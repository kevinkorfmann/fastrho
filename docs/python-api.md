# Python API

The public API is deliberately small. Everything below is available directly from `fastrho`.

## Which function should I call?

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} One VCF contig
:class-card: api-card

`quick_map_from_vcf(...)`

The shortest path from a VCF to a map.
:::

:::{grid-item-card} Many contigs or cohorts
:class-card: api-card

`load_model(...)` + `predict_map_from_vcf(...)`

Load once, then reuse the model.
:::

:::{grid-item-card} Tree sequence or matrix
:class-card: api-card

`predict_map_from_ts(...)` or `predict_map_from_genotype_matrix(...)`

Infer directly from data already in memory.
:::

:::{grid-item-card} Inspect or export
:class-card: api-card

`read_vcf(...)`, `vcf_contigs(...)`, `to_dataframe(...)`, and `write_bed(...)`

Validate input or shape the output without reloading the model.
:::

::::

## One contig, one call

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
    Ne=10_000,
    device="cuda:0",
    input_mode="auto",
    missing="drop-site",
)
```

Signature:

```python
def quick_map_from_vcf(
    vcf_path,
    checkpoint,
    stats,
    *,
    contig=None,
    mutation_rate=1.5e-8,
    Ne=None,
    device="cuda:0",
    as_dataframe=False,
    input_mode="auto",
    missing="drop-site",
):
    ...
```

| Argument | Meaning |
|---|---|
| `checkpoint` | Path to `model.ckpt` from a downloaded or trained model bundle. |
| `stats` | Path to that checkpoint's unchanged `feat_stats.npz` companion archive. It is not derived from the prediction VCF; see {ref}`feat-stats-file`. |
| `contig` | The single chromosome/contig to read. Required for a multi-contig VCF. |
| `mutation_rate` | Per-bp, per-generation mutation rate used as model conditioning. Use a justified value for the organism. |
| `Ne` | Diploid effective size used to convert $\rho$ to absolute $r$. If omitted, the model's auxiliary point estimate is used. |
| `device` | Usually `"cuda:0"`. The Python API does not auto-select a device. |
| `input_mode` | `phased`, `unphased`, `unpolarized`, `raw`, or `auto`. It must match the checkpoint metadata. |
| `missing` | `drop-site` drops any site with a missing genotype; `error` stops instead. |
| `as_dataframe` | Return a pandas table instead of the prediction dictionary. |

## Many chromosomes: load once

```python
import pathlib
import fastrho

vcf = "cohort.vcf.gz"
out = pathlib.Path("maps")
out.mkdir(exist_ok=True)
bundle = pathlib.Path("downloaded-models/domain-randomized-v1")

model, cfg, stats = fastrho.load_model(
    bundle / "model.ckpt",
    bundle / "feat_stats.npz",
    device="cuda:0",
)

for contig in fastrho.vcf_contigs(vcf):
    pred = fastrho.predict_map_from_vcf(
        vcf,
        model,
        cfg,
        stats,
        contig=contig,
        mutation_rate=1.5e-8,
        Ne=10_000,
        device="cuda:0",
        input_mode="auto",
    )
    fastrho.write_bed(
        pred,
        out / f"{contig}.50kb.bed",
        chrom=contig,
        window_size=50_000,
    )
```

Indexed bgzipped VCFs let `cyvcf2` query one contig efficiently. If the VCF header does not contain
`##contig` records, provide your own contig list.

## Tree sequences and in-memory matrices

```python
pred = fastrho.predict_map_from_ts(
    ts, model, cfg, stats,
    mutation_rate=1.5e-8,
    Ne=10_000,
    device="cuda:0",
    input_mode="phased",
)

pred = fastrho.predict_map_from_genotype_matrix(
    gm, positions, model, cfg, stats,
    mutation_rate=1.5e-8,
    Ne=10_000,
    device="cuda:0",
    input_mode="phased",
)
```

For a complete known-answer workflow—including simulation, inference, aligned truth, shape
correlation, absolute-scale bias, and log error—see {doc}`simulation`.

For matrix input:

- `gm` has shape `(n_haplotypes, n_sites)` and values in `{0, 1}`;
- `positions` is a one-dimensional, strictly increasing array in base pairs;
- rows representing the two haplotypes of one diploid must be adjacent for unphased/dosage views;
- missing values must be handled before the call;
- fixed sites are removed automatically.

## Output schema

Every prediction dictionary contains one value per retained interval `[pos_left, pos_right)`:

| Key | Meaning |
|---|---|
| `pos_left`, `pos_right` | 0-based, half-open physical interval edges. |
| `rho_per_bp` | Population-scaled recombination rate $\rho$ per bp. |
| `r_per_bp` | Absolute rate $r=\rho/(4N_e)$ per bp per generation. |
| `rho_ci_lo`, `rho_ci_hi` | Conditional limits for $\rho$. |
| `r_ci_lo`, `r_ci_hi` | Conditional limits for absolute $r$ using `Ne_used`. |
| `Ne_used` | Supplied $N_e$ or the model's auxiliary estimate. |
| `Ne_estimated` | The model's auxiliary point estimate, even when you supply `Ne`. |
| `input_mode`, `contig` | Resolved VCF view and contig for VCF entry points. |

```python
df = fastrho.to_dataframe(pred, chrom="chr1")
df["cM_per_Mb"] = df["r_per_bp"] * 1e8

starts, rates = fastrho.rebin_to_windows(
    pred,
    window_size=50_000,
    key="r_per_bp",
)
```

`df.attrs` retains `Ne_used`, `Ne_estimated`, the coordinate system, and the fact that the absolute
rate interval is conditional on $N_e$.

## Minimal command-line equivalent

```bash
fastrho predict \
  --vcf cohort.vcf.gz \
  --chrom chr1 \
  --checkpoint downloaded-models/domain-randomized-v1/model.ckpt \
  --stats downloaded-models/domain-randomized-v1/feat_stats.npz \
  --mutation-rate 1.5e-8 \
  --ne 10000 \
  --input-mode auto \
  --window-size 50000 \
  --out chr1.50kb.bed
```

The Python API is preferable for many contigs because the model can be loaded once.
