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
  is excluded from the promoted release and is not an active input.
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

Phase 2 is the sole active mosquito analysis. Its promoted figures and generated inputs are copied
into the locked authoritative manuscript snapshot by `reproduce/stage_manuscript.py`. The package
workflow never rewrites manuscript prose, and superseded Phase 3 material is excluded from every
active ledger and gate.
