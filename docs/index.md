# fastrho

**Population genotypes in. A fine-scale recombination map out—without dataset-specific training.**

:::{figure} _static/anim_inference.gif
:alt: Dense SNP tokens pass through the multi-scale stem and an expanded bidirectional Mamba block. Forward and reverse Mamba-2 state scans meet at every token, pass through residual and MLP paths, repeat through the encoder-decoder stack, and produce a mean and variance for every adjacent-SNP interval.
:class: hero-figure
:width: 760px

fastrho builds one feature token per SNP. After a multi-scale local stem, every BiMamba block runs
independent Mamba-2 scans in forward and reverse genomic order. Their states are concatenated and
projected at each token, followed by the scan residual, channel MLP, and second residual. Six
encoder and four decoder blocks—with feature-wise conditioning and encoder skip states—produce a
mean and log variance for every adjacent-SNP interval. The animation expands one schematic
1,024-token context; horizontal spacing shows token order rather than physical distance.
:::

## What the study shows

The central advance of this study is that recombination-map inference can be amortized across
evolutionary contexts. Rather than fitting a new estimator for every population, fastrho learns
from diverse simulated demographies and recombination landscapes and reuses that information to
generate fine-scale maps with uncertainty estimates. The same base model recovered held-out maps
across bottlenecks, expansions, and heterogeneous rate landscapes. Severe canid bottlenecks showed
that demographic history can erase map information from linkage disequilibrium, whereas the
selfing analyses showed that mating-system-specific training can restore the chromosome-wide
pattern. In *Arabis*, comparison with an independent linkage map from 742 $F_2$ offspring recovered
broad agreement for *A. sagittata*; weaker recovery in *A. nemorensis* improved when population
structure was included, although it remained uncertain. Together, these results establish a
scalable framework while defining when specialized simulations are necessary.

The active mosquito analysis uses the freely available Ag1000G Phase 2 AR1 release. One fixed model
produced 45 maps for nine *Anopheles gambiae* and *Anopheles coluzzii* populations across five
chromosome arms. The inferred maps retained the broad expected 2La inversion signal, although the
cross-population relationship between expected arrangement mixing and suppression depth was not
statistically resolved. Independently called Phase 2 laboratory crossovers agreed with the atlas at
broad scale (Spearman $r_s=0.67$ across 32 supported 5-Mb windows), providing a more direct, though
coarse, connection to crossover variation.

Across 15 predefined insecticide-resistance regions, the median focal-to-diversity/H12-matched-control
ratio was 0.71 in *A. gambiae* and 0.72 in *A. coluzzii*. Every population-level median was below one,
but individual regions and populations were heterogeneous. Cold regions also occurred in populations
with weaker local selection signatures, which is compatible with some resistance loci occupying
intrinsically low-recombination regions. The analysis remains observational: selection can distort an
LD-based estimate without changing the underlying crossover rate, and a focal/control ratio below one
does not show that low crossing over caused resistance. The redpoll result provides a complementary
structural example: pooling inversion arrangements created a much stronger cold block than analyzing
either homokaryotype separately.

:::{figure} _static/phase2_anopheles.png
:alt: Phase 2 recombination maps for nine Anopheles populations, with 2La, resistance-region, population, and species-level summaries.
:width: 900px

The open Phase 2 analysis connects population maps, the 2La inversion, 15 resistance regions, and
species-stratified summaries. Population circles and species-median diamonds in panel f show that the
aggregate resistance-region signal is present in both species, while the locus-by-population heatmap
shows why it should not be interpreted as universal.
:::

:::{important}
**`feat_stats.npz` is part of the model—not a file you create from your VCF.** It stores the exact
feature scaling and featurizer metadata required by its matching checkpoint. Download the two files
together, keep them together, and do not adapt the archive for a new cohort. See
{ref}`feat-stats-file` for what it contains, where it comes from, and the retraining case. The
general and specialist checkpoints are all public and checksummed; choose among them using the
model table in {doc}`checkpoints`.
:::

## Choose your path

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Start in 10 minutes
:link: quickstart
:link-type: doc
:class-card: quick-card

Install, inspect one VCF, make a map, and save a BED file.
:::

:::{grid-item-card} Copy the Python recipe
:link: python-api
:link-type: doc
:class-card: quick-card

Use one call for one chromosome or load once for a whole genome.
:::

:::{grid-item-card} Evaluate a known map
:link: simulation
:link-type: doc
:class-card: quick-card

Simulate with `msprime`, infer, align truth, and score shape and absolute scale.
:::

:::{grid-item-card} Check my dataset
:link: your-data
:link-type: doc
:class-card: quick-card

Choose the input view, checkpoint, cohort, filters, and reporting scale.
:::

:::{grid-item-card} Get or reproduce the model
:link: checkpoints
:link-type: doc
:class-card: quick-card

Download verified weights or submit the numbered Betty training workflow.
:::

:::{grid-item-card} Interpret the map
:link: interpretation
:link-type: doc
:class-card: quick-card

Understand $\rho$, absolute $r$, uncertainty, and the biological limits of LD maps.
:::

:::{grid-item-card} Download the paper maps
:link: data
:link-type: doc
:class-card: quick-card

Get the mosquito atlas, *Arabis*, *Arabidopsis*, redpoll, canid, and tree-of-life maps as plot-ready tables.
:::
::::

## The entire workflow

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
    input_mode="auto",
    device="cuda:0",
)

df = fastrho.to_dataframe(pred, chrom="chr1")
fastrho.write_bed(pred, "chr1.50kb.bed", chrom="chr1", window_size=50_000)
```

The result contains population-scaled $\rho$, absolute $r=\rho/(4N_e)$, and a conditional interval
for every retained adjacent-SNP interval. VCF coordinates are converted to **0-based, half-open**
output intervals.

## Designed for broad transfer

- fastrho uses amortized training across diverse simulated demographic histories, so a compatible
  new cohort can be mapped in one inference call without dataset-specific simulation or retraining.
- Each checkpoint declares its supported genotype views and training domain, making model selection
  explicit for phased, unphased, or unpolarized data.
- The model estimates a **population recombination map** from the historical recombination signal in
  linkage disequilibrium, supporting comparisons across genomic regions, populations, and species.
- Results include population-scaled $\rho$ and absolute $r$. Absolute-rate point estimates and
  intervals are conditional on either the supplied $N_e$ or the model's auxiliary point estimate,
  retained as `Ne_used` for transparent reporting.

The associated manuscript is **Scalable inference of recombination maps across evolutionary
contexts**.

```{toctree}
:hidden:
:maxdepth: 1

quickstart
simulation
python-api
your-data
interpretation
checkpoints
data
```
