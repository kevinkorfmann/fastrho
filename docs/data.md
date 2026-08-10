# Paper data

Download the inference products behind the paper without rebuilding the analysis. Every table is
UTF-8, tab-delimited, gzip-compressed, and ready for pandas, R, Polars, or the command line.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} All maps + paper results
:class-card: sd-border-0 sd-shadow-sm

Twelve plot-ready tables, a machine-readable manifest, the active Phase 2 result bundle, and the
committed supporting result snapshots.

{download}`Download the complete bundle (ZIP) <data/downloads/fastrho_paper_data.zip>`
:::

:::{grid-item-card} Stable schema
:class-card: sd-border-0 sd-shadow-sm

Column definitions, units, row counts, source artifacts, sizes, and SHA-256 checksums.

{download}`Download the manifest (JSON) <data/downloads/manifest.json>`
:::

::::

## Reproduce the demographic comparison

The paired competitor benchmark uses exactly the same validated bottleneck and expansion VCFs in
each arm. ReLERNN is trained once with a constant history and once with the generating history;
pyrho is evaluated with constant and matched lookup tables. The frozen raw inputs retain the small
number of malformed genotype rows, while the run manifest records the deterministic validation
applied before all four arms.

{download}`Download frozen inputs <data/downloads/demography_matched_inputs.zip>` ·
{download}`Download paired results <data/downloads/demography_matched_results.json>` ·
{download}`Download every 25-kb prediction <data/downloads/demography_matched_windows.tsv.gz>`

The ordered Slurm workflow, design, environment pins, and analysis scripts are in the repository's
`research/demography_matched` directory.

## Maps

::::{grid} 1 2 2 3
:gutter: 2

:::{grid-item-card} Mosquito atlas
:class-card: sd-border-0 sd-shadow-sm

**41,463 windows · 9 populations · open Phase 2 AR1 data**
*An. gambiae* and *An. coluzzii* across five chromosome arms at 50 kb. Includes $r$, cM/Mb,
$\rho$, cohort metadata, and 2La summaries for both each fixed 40-mosquito map panel and all
eligible released samples.

{download}`Download TSV.gz <data/downloads/anopheles_maps.tsv.gz>`
:::

:::{grid-item-card} *Arabis* cross comparison
:class-card: sd-border-0 sd-shadow-sm

**612 windows · 8 chromosomes**
The F2 map beside separate *A. nemorensis* and *A. sagittata* population maps for the baseline,
small-panel, and structured-selfing campaigns. Values are chromosome-relative.

{download}`Download TSV.gz <data/downloads/arabis_cross_maps.tsv.gz>`
:::

:::{grid-item-card} *Arabidopsis thaliana*
:class-card: sd-border-0 sd-shadow-sm

**1,193 windows · 5 chromosomes**
Selfing-aware fastrho, panmictic pyrho, and the Salomé and Rowan meiotic references at 100 kb.

{download}`Download TSV.gz <data/downloads/arabidopsis_maps.tsv.gz>`
:::

:::{grid-item-card} Redpoll supergene
:class-card: sd-border-0 sd-shadow-sm

**230 windows · chromosome 1**
Pooled and arrangement-stratified maps, disjoint-half replicates, and the established supergene
interval at 500 kb.

{download}`Download TSV.gz <data/downloads/redpoll_maps.tsv.gz>`
:::

:::{grid-item-card} Tree of life
:class-card: sd-border-0 sd-shadow-sm

**6,657 windows · 10 species**
Representative 100-kb inference tracks spanning mammals, insects, and plants. The `status` column
separates external-map validation from repeatability-qualified tracks, and `qualification_tier`
distinguishes the seven core cohorts from three context-limited cohorts.

{download}`Download TSV.gz <data/downloads/tree_of_life_maps.tsv.gz>`
:::

:::{grid-item-card} Canid rescue
:class-card: sd-border-0 sd-shadow-sm

**120 paired regions + one shared landscape**
Own-data versus large-population transfer recovery under a recent bottleneck, plus the map used for
the paper's detailed example.

{download}`Recovery TSV.gz <data/downloads/canid_recovery.tsv.gz>` ·
{download}`Example map TSV.gz <data/downloads/canid_example_map.tsv.gz>`
:::

::::

## Phase 2 manuscript results

The four tables below expose the compact values behind the active mosquito results without requiring
the raw HDF5 files or rerunning inference. The complete Phase 2 ZIP adds the source JSON results,
cohort manifest, release provenance, and map-quality summary.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 2La inversion
:class-card: sd-border-0 sd-shadow-sm

**9 populations**
Arrangement frequencies, expected and observed heterokaryotype frequencies, inside/outside rate
ratios, and suppression depth.

{download}`Download TSV.gz <data/downloads/phase2_2la.tsv.gz>`
:::

:::{grid-item-card} Resistance regions
:class-card: sd-border-0 sd-shadow-sm

**135 locus-population comparisons · 15 regions**
Focal rates, diversity/H12-matched control rates, ratios, local diversity and H12, SNP counts, and
matching depth for every region in every population.

{download}`Download TSV.gz <data/downloads/phase2_resistance.tsv.gz>`
:::

:::{grid-item-card} Laboratory crosses
:class-card: sd-border-0 sd-shadow-sm

**43 common 5-Mb windows**
Detection-aware crossover rates and inferred-map rates after within-arm normalization, including the
support flag used for the broad-scale comparison.

{download}`Download TSV.gz <data/downloads/phase2_pedigree_windows.tsv.gz>`
:::

:::{grid-item-card} pyrho concordance
:class-card: sd-border-0 sd-shadow-sm

**3 frozen regions**
Matched-window Spearman correlations, p-values, and window counts for the direct method comparison.

{download}`Download TSV.gz <data/downloads/phase2_pyrho.tsv.gz>`
:::

::::

{download}`Download the complete Phase 2 result bundle (ZIP) <data/downloads/phase2_results.zip>`

## Plot a map

`pandas` reads the compressed tables directly. This example plots one mosquito cohort and
chromosome arm.

```python
import pandas as pd
import matplotlib.pyplot as plt

maps = pd.read_csv("anopheles_maps.tsv.gz", sep="\t")
track = maps.query("cohort == 'gamb_BF' and chromosome_arm == '2L'")

plt.plot(track.start_bp / 1e6, track.cM_per_Mb, lw=1)
plt.xlabel("Position on 2L (Mb)")
plt.ylabel("Recombination rate (cM/Mb)")
plt.show()
```

For *Arabis*, switch among `baseline_selfing`, `small_panel_selfing`, and `structured_selfing` with
the `campaign` column. For the tree-of-life table, group by `species_key`; blank reference values
mean that the track is an inference rather than a reference-map validation.

## Data contract

- Genomic `start_bp` and `end_bp` coordinates are 0-based and half-open.
- Blank cells are missing values, not zeros.
- `rate_per_bp` is the per-generation recombination probability per base; `rho_per_bp` is
  population-scaled. The *Arabis* table is explicitly chromosome-relative with mean 1 per series.
- In `anopheles_maps.tsv.gz`, columns prefixed with `panel_` describe the exact 40-mosquito panel
  used for map inference; columns prefixed with `full_` use all eligible released Phase 2 samples.
- `phase2_resistance.tsv.gz` reports observational LD-based estimates. A focal/control ratio below
  one does not by itself establish a lower meiotic crossover rate or a causal role in resistance.
- `phase2_results.zip` is the authoritative compact result bundle for the active mosquito analysis.
  The complete paper ZIP also retains supporting benchmark snapshots under `results/`; restricted
  MalariaGEN Phase 3 material is not distributed.
- Regenerate every download with `python scripts/export_paper_data.py`.
