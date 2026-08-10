# Phase 2 manuscript migration log

## Frozen design

- Release: Ag1000G Phase 2 AR1, AgamP4 coordinates.
- Raw inputs: five wild-haplotype HDF5 files, four autosomal cross SHAPEIT
  haplotype/sample pairs, and release metadata; all are recorded in the remote
  SHA-256 manifest.
- Primary cohorts: four *Anopheles coluzzii* and five *Anopheles gambiae*
  species-by-population cohorts. GM, GW, and KE are excluded because the Phase
  2 resource paper reports uncertain species status.
- Panel: 40 diploids per cohort, NumPy seed 2026, sampled from the 1,058
  identifiers present in all five chromosome-arm HDF5 files. The all-arm
  intersection was introduced after the first extraction attempt identified a
  selected BFcol mosquito absent from the X release. The superseded extraction
  remains recoverable on Sesame and is not eligible for promotion.
- Model: unchanged broadened-
  \(N_e\) checkpoint and feature statistics used by the frozen mosquito atlas;
  mutation rate \(3.5\times10^{-9}\).

## Release-specific analyses

- 2La: all 202 MalariaGEN tag SNPs match the released 2L haplotypes. Arrangement
  frequency uses all eligible release samples in each primary cohort; map
  suppression uses the frozen 40-diploid panels.
- Resistance: the literature-frozen 6-, 8-, and 15-region panels are retained.
  Primary controls match same-arm position, SNP density, nucleotide diversity,
  and H12. Per-locus, leave-one-locus, and overlap-de-duplicated summaries are
  derived from the same matched ratios.
- Pyrho: three representative cohorts use identical frozen 20-haplotype
  subsamples on 3R:6--14 Mb. This is an estimator comparison, not direct
  crossover validation.
- Crosses: all 11 released Phase 2 crosses are used. Calls operate on released
  SHAPEIT haplotypes, not later genotype-quality/site-filter arrays. There is no
  held-out-family endpoint. SHAPEIT values above one (observed codes 2 and 255)
  are normalized to missing. The promoted comparison is restricted to
  chromosome-arm-normalized 5-Mb spatial variation, with an exact nonzero
  within-arm circular-shift null.

## Manuscript isolation

Every mosquito-dependent passage in `main.tex` and `si.tex` is enclosed by an
explicit `ANOPHELES_SLOT` comment pair. Phase 3 and Phase 2 provide exactly the
same 13 fragment names. Switching replaces only slot contents and
variant-owned generated assets. The common-text audit reconstructs all frozen
Phase 3 fragments in the current manuscript, removes slot comments, normalizes
whitespace, and requires equality with the pre-migration manuscript. Phase 2
activation remains disabled until all result, figure, table, provenance,
compilation, and restoration gates pass.
