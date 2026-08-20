# *Anopheles* fine-scale recombination atlas

Population-resolved fine-scale recombination maps for **13 cohorts** of three African malaria-vector
species — *Anopheles gambiae*, *An. coluzzii*, and *An. arabiensis* — across all five chromosome arms
(**2R, 2L, 3R, 3L, X**). Inferred directly from Ag1000G (Ag3.0) whole-genome haplotypes with the
single frozen `high-ne-v1` checkpoint, without per-population retraining or lookup tables. Long
sequences were evaluated in overlapping 1,024-token contexts and stitched into continuous maps.

To our knowledge this is the first fine-scale recombination atlas resolved at the level of individual
*Anopheles* populations. Because these species lack PRDM9, the fine-scale landscape is largely
conserved across cohorts; the population-level differences are driven mainly by segregating
chromosomal inversions (e.g. 2La), whose recombination-suppressing effect was directionally
associated with the local inversion frequency.

## Files

- `bed/<cohort>.bed` — one file per cohort, all five arms concatenated, one **50-kb window** per line:

  | column | meaning |
  |---|---|
  | `chrom` | chromosome arm (AgamP4) |
  | `start`, `end` | 0-based half-open window (bp) |
  | `rate_per_bp` | recombination rate $r$ (per bp, per generation) |
  | `cM_per_Mb` | the same rate as centimorgans per megabase ($r\times10^{8}$) |
  | `rho_per_bp` | population-scaled rate $\rho = 4N_e r$ (per bp) |

- `manifest.tsv` — per-cohort metadata: species, country, approximate sampling lat/lon, sample size,
  SNP count, estimated $N_e$, and the 2La inversion frequency $p$ and heterokaryotype frequency
  $H = 2p(1-p)$ for the exact 40-mosquito map panel. These are not the full-cohort estimates used
  in the atlas analysis.

Cohort codes are `<species>_<country>` (e.g. `gamb_BF` = *An. gambiae*, Burkina Faso). A 100-kb
resolution is also available in the source map files.

## Reproduce

The committed NPZ maps and this release manifest are the inputs to
`scripts/export_paper_data.py`, which regenerates the public combined table and self-documented ZIP.
See the paper's *Anopheles* section and the
[dataset guide](../../../../../docs/your-data.md) for the current model-selection and reporting
contract.

## Citation

If you use these maps, please cite the `fastrho` paper (see the repository root). Underlying genome
data: the *Anopheles gambiae* 1000 Genomes Consortium (Ag1000G / Ag3.0).
