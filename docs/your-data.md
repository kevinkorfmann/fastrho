# Use fastrho with your dataset

The most important choice is not a tuning parameter. It is whether your data and checkpoint describe
the same **genotype view and population-genetic regime**.

## Choose the input view

| Your data | `input_mode` | Required checkpoint support |
|---|---|---|
| Phased and ancestrally polarized haplotypes | `phased` | `phased` |
| Unphased diploid genotypes with trusted allele orientation | `unphased` | `unphased` |
| Unphased and no trustworthy ancestral allele | `unpolarized` | `unpolarized` |

For a VCF, `auto` distinguishes phased (`|`) from unphased (`/`) GT calls. It cannot decide whether
alleles are polarized. Never feed unphased data to a phased-only checkpoint or silently treat an
unknown reference allele as ancestral.

The machine-readable registry included with the package, `fastrho/model_registry.json`, currently
declares one released general model and two planned specialist artifacts:

| Model | Intended use | Views | Status |
|---|---|---|---|
| `domain-randomized-v1` | Primary general model | phased, unphased, unpolarized | `available` |
| `selfing-v1` | Strong-selfing regime | unpolarized | `artifact-pending` |
| `high-ne-v1` | High-$N_e$ regime | phased | `artifact-pending` |

The numerical ranges in a model's training manifest are a qualification domain, not a guarantee for
every dataset inside those ranges.

## Use the different models

The Python API is identical for every model. Choose a biological regime, keep its checkpoint and
statistics archive together, and pass an input view that the registry declares:

:::{note}
Only `domain-randomized-v1` is currently downloadable. The specialist rows document intended
selection behavior for artifacts whose registry status is still `artifact-pending`.
:::

| Start with | Use it for | Set `input_mode` to |
|---|---|---|
| `domain-randomized-v1` | General outbred transfer across diverse demographic histories | `phased`, `unphased`, or `unpolarized` |
| `selfing-v1` | Predominantly selfing populations | `unpolarized` |
| `high-ne-v1` | High-diversity, high-$N_e$ cohorts | `phased` |

```python
from pathlib import Path
import fastrho

model_id = "domain-randomized-v1"  # selfing-v1 and high-ne-v1 are not released yet
bundle = Path("downloaded-models") / model_id
input_mode = {
    "domain-randomized-v1": "unpolarized",  # phased/unphased are also supported
    "selfing-v1": "unpolarized",
    "high-ne-v1": "phased",
}[model_id]

pred = fastrho.quick_map_from_vcf(
    "cohort.vcf.gz",
    bundle / "model.ckpt",
    bundle / "feat_stats.npz",
    contig="chr1",
    mutation_rate=1.5e-8,
    Ne=10_000,
    input_mode=input_mode,
    device="cuda:0",
)
```

For the general model, select the input view that describes the data; `auto` can distinguish VCF
phase separators but cannot establish ancestral polarization. For a specialist model, change the
bundle and `input_mode` together. The model ID is a provenance label—the current API deliberately
loads explicit local paths so checkpoint/statistics pairing remains visible and reproducible.

## VCF checklist

Before inference, make sure the selected cohort has:

- one biological population rather than a mixture of strongly structured groups;
- diploid GT calls in a plain or bgzipped VCF;
- one contig per prediction call;
- single-base, biallelic SNPs in increasing physical order;
- at least two segregating, complete SNPs after filtering;
- a sample size compatible with the checkpoint's training metadata;
- a justified per-generation mutation rate;
- a justified diploid $N_e$ if absolute $r$ matters.

`read_vcf` skips indels and multiallelic records. With `missing="drop-site"`, a site is removed if
**any** included sample is missing; no reference imputation occurs. If this removes too much data,
perform and report a principled cohort-level missingness filter before fastrho rather than changing
missing calls to zero.

## Cohort design matters

An LD map is a property of the sampled population history as well as the underlying crossover
landscape. Before creating a cohort:

1. use PCA, relatedness, geography, and metadata to define a coherent sample;
2. avoid combining species, deeply separated populations, or opposite inversion arrangements unless
   that mixture is the scientific target;
3. record every sample exclusion and whether the analysis used phased, unphased, or folded data;
4. use the same cohort definition when comparing maps.

Recent bottlenecks can erase fine-scale information. Selfing changes effective recombination.
Inversions, selection, admixture, and relatedness can create long-range LD that resembles suppressed
crossing over. Choose a specialized checkpoint only when its training regime matches the mechanism
you intend to model.

## Pick a reporting scale

fastrho predicts adjacent-SNP intervals. For comparisons and plots, aggregate by physical span:

```python
fastrho.write_bed(pred, "map.25kb.bed", chrom="chr1", window_size=25_000)
fastrho.write_bed(pred, "map.100kb.bed", chrom="chr1", window_size=100_000)
```

Use the same windows, coordinate assembly, and accessible sequence mask across populations. A finer
grid is not automatically more informative: sparse SNPs, bottlenecks, and model transfer can lower
the effective resolution.

## Common errors

| Error | What to do |
|---|---|
| `VCF contains multiple contigs` | Pass `contig="..."` or loop over `vcf_contigs(...)`. |
| `no biallelic SNP records found` | Check the contig name, GT field, filters, and whether records are SNPs. |
| `missing genotype` | Use `drop-site` or filter missingness before inference; do not impute reference silently. |
| `positions ... strictly increasing` | Sort records and remove duplicate positions. |
| `stats were trained with ... tokens` | Choose the matching `input_mode` and checkpoint/statistics pair. |
| `model archive has no ... view` | The checkpoint is incompatible with that genotype view. |
| CUDA or Mamba import failure | Rebuild the inference environment against a supported PyTorch/CUDA combination. |

## Minimum reporting record

For a map someone else can reproduce, save:

- fastrho version and Git commit;
- checkpoint, statistics archive, model ID, and SHA-256 hashes;
- reference assembly, VCF version, cohort/sample manifest, and contig list;
- filtering, phase, polarization, `input_mode`, and missing-data policy;
- mutation rate, supplied or estimated $N_e$, and window size;
- the exact command or Python script that produced the map.

See {doc}`interpretation` before treating a population map as a meiotic crossover map.
