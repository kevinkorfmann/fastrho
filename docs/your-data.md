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

The machine-readable registry is
[`fastrho/model_registry.json`](https://github.com/kevinkorfmann/fastrho/blob/main/fastrho/model_registry.json).
It declares six public, checksummed releases:

| Model | Intended use | Views | Status |
|---|---|---|---|
| `domain-randomized-v1` | Primary general model | phased, unphased, unpolarized | `available` |
| `base-v1` | Ordinary-demography specialist | phased | `available` |
| `composite-ld-v1` | Folded composite-LD specialist | unpolarized | `available` |
| `dog-bottleneck-v1` | Canine bottleneck specialist | unpolarized | `available` |
| `selfing-v1` | Strong-selfing regime | phased | `available` |
| `high-ne-v1` | High-$N_e$ regime | phased | `available` |

The numerical ranges in a model's training manifest are a qualification domain, not a guarantee for
every dataset inside those ranges.

## Use the different models

The Python API is identical for every model. Choose a biological regime, keep its checkpoint and
statistics archive together, and pass an input view that the registry declares:

| Start with | Use it for | Set `input_mode` to |
|---|---|---|
| `domain-randomized-v1` | General outbred transfer across diverse demographic histories | `phased`, `unphased`, or `unpolarized` |
| `base-v1` | Ordinary-demography, polarized haplotypes | `phased` |
| `composite-ld-v1` | Unphased and unpolarized diploid genotypes | `unpolarized` |
| `dog-bottleneck-v1` | Canine cohorts matching the bottleneck prior | `unpolarized` |
| `selfing-v1` | Predominantly selfing populations | `phased` |
| `high-ne-v1` | High-diversity, high-$N_e$ cohorts such as *Anopheles* | `phased` |

```python
from pathlib import Path
import fastrho

model_id = "domain-randomized-v1"
bundle = Path("downloaded-models") / model_id
input_mode = {
    "domain-randomized-v1": "unpolarized",  # phased/unphased are also supported
    "base-v1": "phased",
    "composite-ld-v1": "unpolarized",
    "dog-bottleneck-v1": "unpolarized",
    "selfing-v1": "phased",
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

(sample-count-contract)=
## Samples: what is read and how many to use

For an ordinary diploid VCF, a **sample** means one column after `FORMAT`. fastrho reads every such
column in the selected VCF; neither `read_vcf`, `predict_map_from_vcf`, nor `quick_map_from_vcf` has
an internal sample limit or sampling step. Each diploid GT call is expanded into two adjacent allele
rows, so $N$ VCF samples produce a matrix with $2N$ rows. The two rows remain haplotypes in `phased`
mode and are paired back into $N$ dosages in `unphased` or `unpolarized` mode.

This has two practical consequences:

1. A 1,000-sample VCF is **not** automatically reduced to the model's training range. All 1,000
   samples are used and the model is conditioned on 2,000 haplotypes.
2. Sample selection must happen before inference. Create a population-specific VCF with a tool such
   as `bcftools view -S cohort.samples ...`, or construct and subset the genotype matrix yourself.
   Choose samples using population membership, relatedness, geography, sequencing quality, and the
   scientific question—not merely the first rows in the file.

The current released checkpoint envelopes are:

| Checkpoint | Training sample envelope | VCF interpretation |
|---|---:|---|
| `domain-randomized-v1` | 20--200 haplotypes | 10--100 diploid samples |
| `base-v1` | 20--200 haplotypes | 10--100 diploid samples; phased |
| `composite-ld-v1` | 20--200 haplotypes | 10--100 diploid samples; unpolarized |
| `high-ne-v1` | 20--200 haplotypes | 10--100 diploid samples; phased |
| `dog-bottleneck-v1` | 60--134 haplotypes | 30--67 diploid samples |
| `selfing-v1` | 50--200 inbred lines | one retained haplotype per near-homozygous line; not the ordinary two-rows-per-diploid interpretation |

These are qualification envelopes, not promises of accuracy and not targets that justify padding a
small cohort or arbitrarily discarding population structure. The code's technical lower bound is
only two allele rows, but that is not a scientifically useful minimum for LD inference. For the
general model, treat 10 diploids as the edge of its training domain and prefer at least 20 unrelated
diploids when available. Validate a new design with regime-matched simulations and maps from
disjoint or repeated subsets. If a cohort is larger than the checkpoint range, analyze documented,
coherent subsets within the range and assess agreement rather than sending the full panel through
out of distribution.

You can make the count explicit before loading the GPU model:

```python
gm, positions, meta = fastrho.read_vcf(
    "cohort.vcf.gz",
    contig="chr1",
    return_metadata=True,
)
n_diploid = gm.shape[0] // 2
print(f"{n_diploid} diploid samples -> {gm.shape[0]} allele rows")

if not 10 <= n_diploid <= 100:
    raise ValueError(
        "cohort is outside the 10--100 diploid training envelope of "
        "domain-randomized-v1; define and validate a suitable cohort first"
    )
```

## VCF checklist

Before inference, make sure the selected cohort has:

- one biological population rather than a mixture of strongly structured groups;
- diploid GT calls in a plain or bgzipped VCF;
- one contig per prediction call;
- single-base, biallelic SNPs in increasing physical order;
- at least two segregating, complete SNPs after filtering;
- a deliberately selected sample count within the checkpoint envelope—fastrho will not subsample;
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

## Start from a manuscript species

The repository includes a machine-readable preset and source route for every active empirical
analysis. The comparative SI panel is a fixed set of ten species (seven core and three
context-limited cohorts). Dedicated presets cover dog, wolf, the two Phase 2 *Anopheles* species,
and redpoll. Restricted Phase 3 analyses and presets are not distributed in this repository.

```bash
python3 examples/manuscript_species/data.py list
python3 examples/manuscript_species/data.py show human
python3 examples/manuscript_species/data.py download human --dry-run --include-companions
```

The accompanying runner resolves the documented contig, mutation rate, genotype view, model
profile, and reporting window while still requiring you to supply the checkpoint explicitly:

```bash
python3 examples/manuscript_species/infer.py \
  --species human --vcf cohort.chr2.vcf.gz \
  --checkpoint downloaded-models/domain-randomized-v1/model.ckpt \
  --stats downloaded-models/domain-randomized-v1/feat_stats.npz \
  --out map.chr2.bed --dry-run
```

See the [complete manuscript-species guide](https://github.com/kevinkorfmann/fastrho/tree/main/examples/manuscript_species)
for direct download commands, deterministic Phase 2 cohort extraction, the fixed 10-species SI
panel, and its dedicated Phase 2 presets. Source data and generated maps stay outside Git; the
manifest and utilities that reproduce them are tracked.
