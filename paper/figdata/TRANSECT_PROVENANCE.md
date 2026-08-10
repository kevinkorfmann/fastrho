# Cross-species recombination comparison: data provenance

The displayed comparison contains ten retained public genotype panels: seven core cohorts and
three context-limited cohorts. One representative chromosome is shown per species, except that the
external-map score for *Arabidopsis thaliana* is averaged across chromosomes 1–5. The retained set,
statistics, tracks, and evidence tier are serialized in `transect.json`; the same-panel pyrho tracks
are serialized in `transect_pyrho.json`.

| Key | Cohort and source | Samples | Analyzed chromosome | Evidence tier | Reported comparison |
|---|---|---:|---|---|---|
| `human` | IGSR 1000 Genomes phase 3, CEU (`20130502`) | 99 diploids | 2 | core | HapMap external map |
| `cattle` | EBI NextGen UGBT, Uganda | 25 diploids | 1 | core | split-sample and pyrho |
| `sheep` | EBI NextGen IROA, Iran | 20 diploids | 1 | core | split-sample and pyrho |
| `goat` | EBI NextGen IRCH, Iran | 20 diploids | 1 | core | split-sample and pyrho |
| `donkey` | Todd et al. 2022, Chinese cohort (`PRJEB55549`) | 51 diploids | `CM027690.1` | context-limited | split-sample and pyrho |
| `dmel` | Drosophila Genetic Reference Panel | 205 lines | 2L | core | Comeron external map |
| `jewelwasp` | *Nasonia vitripennis* reference panel (`PRJEB33514`) | 34 lines | `NC_015868.2` | context-limited | split-sample and pyrho |
| `athal` | 1001 Genomes, south-Swedish sample | 78 accessions | 1–5 | core | Salomé external map |
| `aspen` | SwAsp (`PRJEB79788`) | 99 diploids | chr1 | core | split-sample and pyrho |
| `chestnut` | European Variation Archive (`PRJEB87510`) | 97 diploids | Chr1 | context-limited | split-sample and pyrho |

External-map comparisons are Pearson correlations in aligned 100-kb windows. For species without
an external fine-scale map, split-sample correlation measures repeatability rather than accuracy;
agreement with pyrho compares two LD-based estimators applied to the same genotype panel. Donkey
and Chinese chestnut combine sampling localities, and the jewel-wasp panel consists of inbred lines,
so those cohorts are marked context-limited in every generated artifact.

Candidate panels not retained in the comparison remain identified in
`examples/manuscript_species/species.json` with the role `excluded: cohort-design review`. They are
not included in the figure, the download table, or the figure-input manifest.
