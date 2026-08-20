# Inferred maps and paper data

Download the recombination landscapes and compact results behind the paper without rebuilding the
analysis. Every table is UTF-8, tab-delimited, gzip-compressed, and ready for pandas, R, Polars, or
the command line.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} All maps + paper results
:class-card: sd-border-0 sd-shadow-sm

Eight plot-ready tables, a machine-readable manifest, and committed supporting result snapshots.

{download}`Download the complete bundle (ZIP) <data/downloads/fastrho_paper_data.zip>`
:::

:::{grid-item-card} Stable schema
:class-card: sd-border-0 sd-shadow-sm

Column definitions, units, row counts, source artifacts, sizes, and SHA-256 checksums.

{download}`Download the manifest (JSON) <data/downloads/manifest.json>`
:::

::::

## Inferred-map downloads

The complete ZIP contains every table below. For one analysis, download its TSV.gz directly; for
the Ag3 mosquito atlas, use the self-documented ZIP. The scale is part of each column name and
is summarized here before the detailed descriptions.

| Analysis | Download | Native scale | Use directly for |
|---|---|---|---|
| Ag3.0 mosquitoes | [ZIP][anopheles-zip] · [TSV.gz][anopheles-tsv] | `rho_per_bp`; also $N_e$-conditional `rate_per_bp` and `cM_per_Mb` | Population-scaled or conditional absolute plots; retain `Ne_used` with absolute values |
| *Arabis* | [TSV.gz][arabis-tsv] | Chromosome-relative rate, mean 1 within each map series | Comparing spatial shape among population and $F_2$ maps |
| *Arabidopsis thaliana* | [TSV.gz][arabidopsis-tsv] | Per-generation rate per bp | Comparing inferred and meiotic-reference map shape at 100 kb |
| Redpoll | [TSV.gz][redpoll-tsv] | `rho_per_bp` | Comparing pooled and arrangement-specific population-scaled maps |
| Ten-species comparison | [TSV.gz][tree-of-life-tsv] | Per-generation rate per bp | Plotting 100-kb tracks within a species; normalize within species for cross-species shape comparisons |
| Canid simulation example | [TSV.gz][canid-example-tsv] | Per-generation rate per bp | Comparing the known simulation input with large- and bottlenecked-population inference |
| Canid recovery benchmark | [TSV.gz][canid-recovery-tsv] | 100-kb log-rate correlation | Comparing own-data and large-population transfer recovery across 120 paired regions |
| Demography-matched benchmark | [TSV.gz][demography-windows-tsv] | Per-generation rate per bp | Replotting every matched fastrho, pyrho, and ReLERNN prediction on the common 25-kb grid |

[anopheles-zip]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/anopheles_maps.zip
[anopheles-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/anopheles_maps.tsv.gz
[arabis-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/arabis_cross_maps.tsv.gz
[arabidopsis-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/arabidopsis_maps.tsv.gz
[redpoll-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/redpoll_maps.tsv.gz
[tree-of-life-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/tree_of_life_maps.tsv.gz
[canid-example-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/canid_example_map.tsv.gz
[canid-recovery-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/canid_recovery.tsv.gz
[demography-windows-tsv]: https://github.com/kevinkorfmann/fastrho/raw/refs/heads/main/docs/data/downloads/demography_matched_windows.tsv.gz

`rho_per_bp` means the diploid population-scaled rate $\rho=4N_e r$ per bp. A `rate_per_bp`
column means the per-generation recombination probability $r$ per bp; multiply it by $10^8$ to
obtain cM/Mb. Columns ending in `_relative_rate` are dimensionless and must not be converted to
cM/Mb. Only the mosquito table records the $N_e$ used for its absolute conversion. Treat inferred
per-bp values in the *Arabidopsis* and ten-species tables as the archived paper scale for
within-track comparisons, not as independently calibrated cross-species absolute rates. The
[interpretation guide](interpretation.md) explains which scale to report.

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

:::{grid-item-card} Mosquito recombination landscapes
:class-card: sd-border-0 sd-shadow-sm

**59,940 windows · 13 populations · MalariaGEN Ag3.0 data**
Six *An. gambiae*, four *An. coluzzii*, and three *An. arabiensis* populations across five AgamP4
chromosome arms at 50 kb. Includes population-scaled $\rho$, conditional absolute $r$ and cM/Mb,
the arm-specific $N_e$ used for that conversion, cohort/species metadata, and fixed-panel 2La
summaries.

{download}`Download self-documented ZIP <data/downloads/anopheles_maps.zip>` ·
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

## Mosquito-map scale and $N_e$: read before analysis

The mosquito table contains **LD-based population recombination maps**, not a direct record
of contemporary crossovers. Each row is one 0-based, half-open 50-kb AgamP4 window (the terminal
window of an arm can be shorter). The fixed inference panel contains 40 diploid mosquitoes, or 80
haplotypes, per cohort. The table includes *An. gambiae*, *An. coluzzii*, and *An. arabiensis*, 13
populations, and arms 2R, 2L, 3R, 3L, and X.

| Column | Scale and interpretation |
|---|---|
| `rho_per_bp` | Population-scaled rate $\rho$ per bp; this is the model-scale map and has not been divided by $N_e$. |
| `Ne_used` | Arm-specific auxiliary model point estimate used to convert $\rho$ to absolute $r$; it is not an independently validated demographic or census estimate. |
| `rate_per_bp` | Per-generation recombination probability per bp, already computed as $\rho/(4N_{e,\mathrm{used}})$. |
| `cM_per_Mb` | The same pre-scaled absolute value in cM/Mb, computed as `rate_per_bp * 1e8`. |
| `panel_*` | 2La summaries for the exact fixed 40-mosquito panel used to infer the maps. |

Choose the scale that matches the analysis:

- For hotspots, cold regions, or other spatial comparisons within a population, use
  `rho_per_bp`, a within-population ratio, or a map normalized to mean 1 within each chromosome
  arm. These analyses do not require an absolute $N_e$.
- For comparisons of map shape among populations, normalize each population separately. Raw
  differences in $\rho$ can reflect differences in $N_e$ as well as differences in recombination.
- For absolute rates, use `rate_per_bp` or `cM_per_Mb` only as values conditional on `Ne_used`, and
  retain that value in tables and figure metadata.

`Ne_used` comes from the checkpoint's auxiliary region-level $N_e$ head. The head was trained on
simulations with known $N_e$ and uses diversity-informative features including SNP spacing, allele
frequencies, local nucleotide diversity, and haplotype richness. For these maps it also assumes a
mutation rate of $3.5\times10^{-9}$ per bp per generation. This makes `Ne_used` a model-based,
diversity-informed estimate, not an external validation of demographic history. The cross-based
comparison evaluated normalized spatial shape and therefore did not validate $N_e$ or the absolute
cM/Mb scale.

**No additional $N_e$ rescaling is needed to plot or use the released `rate_per_bp` or
`cM_per_Mb` values. Do not divide them by $N_e$ again.** Absolute values are conditional on the
reported `Ne_used`. If you have an independently justified value $N_{e,\mathrm{target}}$, replace
the absolute columns—do not rescale them indirectly—with:

```python
maps["rate_per_bp_external_Ne"] = maps["rho_per_bp"] / (4 * Ne_target)
maps["cM_per_Mb_external_Ne"] = maps["rate_per_bp_external_Ne"] * 1e8
```

Keep `rho_per_bp` unchanged. Record the external $N_e$, its source, and the diploid convention
$\rho=4N_e r$ with any rescaled result. For comparisons focused on spatial shape, state whether
tracks were normalized within each arm; for absolute comparisons, propagate uncertainty in $N_e$.

A Watterson estimate can provide a useful sensitivity value when no external demographic estimate
is available,

$$
\widehat N_{e,W}=\frac{S}{4\mu L_{\mathrm{callable}}a_{n-1}},\qquad
a_{n-1}=\sum_{i=1}^{n-1}\frac{1}{i},
$$

where $S$ is the number of segregating sites, $n$ is the number of sampled haplotypes, and
$L_{\mathrm{callable}}$ is the callable rather than the physical sequence length. Use variant and
missing-data filters consistent with the map input and report the assumed mutation rate. Because
selection, structure, bottlenecks, inversions, and errors in callable length can shift this estimate,
do not substitute it as uniquely correct. Instead, recompute the absolute columns under the
Watterson estimate and other defensible $N_e$ values and report the resulting sensitivity range.
Where a direct pedigree or linkage map exists, chromosome-wide calibration against that map is
preferable to relying on a diversity estimate alone.

For *Arabis*, switch among `baseline_selfing`, `small_panel_selfing`, and `structured_selfing` with
the `campaign` column. For the tree-of-life table, group by `species_key`; blank reference values
mean that the track is an inference rather than a reference-map validation.

## Data contract

- Genomic `start_bp` and `end_bp` coordinates are 0-based and half-open.
- Blank cells are missing values, not zeros.
- `rate_per_bp` is the per-generation recombination probability per base; `rho_per_bp` is
  population-scaled. In the mosquito table, `Ne_used` is arm-specific and exactly identifies the
  scale conversion represented by each row. The *Arabis* table is explicitly chromosome-relative
  with mean 1 per series.
- In `anopheles_maps.tsv.gz`, columns prefixed with `panel_` describe the exact 40-mosquito Ag3.0
  panel used for map inference.
- The complete paper ZIP also retains supporting benchmark snapshots under `results/`.
- Regenerate every download with `python scripts/export_paper_data.py`.
