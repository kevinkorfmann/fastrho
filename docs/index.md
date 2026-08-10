# fastrho

**Population genotypes in. A fine-scale recombination map out, without dataset-specific training.**

:::{figure} _static_public/fastrho_schematic.png
:alt: Method schematic showing SNP feature tokens, bidirectional state-space inference, and interval-level recombination-rate output.
:class: hero-figure
:width: 960px

Each SNP contributes one feature token for the interval on its right. A multi-scale local stem and
bidirectional Mamba-2 encoder-decoder integrate linkage-disequilibrium context in both genomic
directions, then return a mean and dispersion for population-scaled recombination rate at every
retained interval.
:::

:::{note}
fastrho is a public research alpha. Interfaces and model recommendations may evolve; pin the
software revision and model checksums used for an analysis.
:::

## Start here

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
:class-card: quick-card

Install fastrho, inspect one VCF, make a map, and save a BED file.
:::

:::{grid-item-card} Python API
:link: python-api
:link-type: doc
:class-card: quick-card

Use one call for one chromosome or load once for a whole genome.
:::

:::{grid-item-card} Known-answer simulation
:link: simulation
:link-type: doc
:class-card: quick-card

Simulate an explicit rate map, align truth, and score shape and scale.
:::

:::{grid-item-card} Check your dataset
:link: your-data
:link-type: doc
:class-card: quick-card

Choose the input view, cohort, checkpoint, filters, and reporting scale.
:::

:::{grid-item-card} Get the model bundle
:link: checkpoints
:link-type: doc
:class-card: quick-card

Download and verify the checkpoint and its required companion archive.
:::

:::{grid-item-card} Interpret the map
:link: interpretation
:link-type: doc
:class-card: quick-card

Understand rho, absolute rate, uncertainty, and the limits of LD maps.
:::

:::{grid-item-card} Phase 2 maps and results
:link: data
:link-type: doc
:class-card: quick-card

Download the nine-population *Anopheles* maps and plot-ready result tables.
:::
::::

## Minimal workflow

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
    missing="drop-site",
    device="cuda:0",
)

df = fastrho.to_dataframe(pred, chrom="chr1")
fastrho.write_bed(pred, "chr1.50kb.bed", chrom="chr1", window_size=50_000)
```

The result contains population-scaled rho, absolute rate conditional on `Ne_used`, and a model
interval for each retained adjacent-SNP interval. VCF coordinates are converted to **0-based,
half-open** output intervals.

## Scientific scope

fastrho amortizes training across simulated demographic histories and rate landscapes, allowing a
compatible new cohort to be analyzed without fitting a new estimator. Compatibility still matters:
the checkpoint must support the genotype view and cover a plausible biological regime.

An inferred LD map reflects population history as well as crossover rate. Bottlenecks, structure,
inversions, selection, relatedness, selfing, and gene conversion can alter recoverable map shape or
scale. Use realistic simulations or independent crossover information before making a biological
claim, and describe the result as a population recombination map unless direct evidence supports a
meiotic interpretation.

## Open Phase 2 analysis

The active mosquito analysis uses the freely available Ag1000G Phase 2 AR1 release: nine
populations, five chromosome arms, and a predefined panel of 15 resistance regions. Ready-to-use
maps, result tables, and provenance are available on the {doc}`data` page.

:::{figure} _static_public/phase2_anopheles.png
:alt: Phase 2 Anopheles recombination maps, inversion comparison, and resistance-region summaries.
:width: 100%
:::

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
