# Phase 2 maps and results

The active *Anopheles* manuscript analysis uses the freely available MalariaGEN Ag1000G Phase 2
AR1 release. It contains five *Anopheles gambiae* and four *Anopheles coluzzii* population panels,
each represented by the same 40 diploid mosquitoes across chromosome arms 2R, 2L, 3R, 3L, and X.
Coordinates use the AgamP4 assembly.

## Downloads

| File | Contents |
|---|---|
| [anopheles_maps.tsv.gz](data/downloads/anopheles_maps.tsv.gz) | All 41,463 non-overlapping map windows, with absolute and population-scaled rates and cohort metadata |
| [phase2_2la.tsv.gz](data/downloads/phase2_2la.tsv.gz) | Population-level 2La arrangement frequencies and inferred suppression summaries |
| [phase2_resistance.tsv.gz](data/downloads/phase2_resistance.tsv.gz) | Focal and matched-control rates for the predefined 15-region resistance panel |
| [phase2_pedigree_windows.tsv.gz](data/downloads/phase2_pedigree_windows.tsv.gz) | Broad-scale crossing-over and inferred-map comparison windows |
| [phase2_pyrho.tsv.gz](data/downloads/phase2_pyrho.tsv.gz) | Matched-subsample pyrho comparisons for three representative regions |
| [phase2_results.zip](data/downloads/phase2_results.zip) | Compact source JSON, cohort manifest, provenance, and pedigree-window table |

The complete ready-to-use BED atlas is also committed under
[`paper/anopheles_variants/phase2/release/atlas_anopheles/`](https://github.com/kevinkorfmann/fastrho/tree/main/paper/anopheles_variants/phase2/release/atlas_anopheles).
The scripts, frozen cohort selection, maps, figures, result files, and checksum manifests needed to
audit the analysis are in the surrounding Phase 2 module.

## Interpretation

These are LD-based population recombination maps. A low inferred focal-to-control ratio does not by
itself establish a lower meiotic crossover rate or a causal role in resistance. Selection can alter
local diversity and LD, while resistance loci may also lie in regions with intrinsically low
recombination. The matched-control analysis accounts partly for diversity and H12, and the
cross-population comparison shows how consistently each pattern is recovered across backgrounds.

Restricted MalariaGEN Phase 3 analyses, results, maps, and manuscript drafts are not distributed in
this public repository.
