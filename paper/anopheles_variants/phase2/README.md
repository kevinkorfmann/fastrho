# Ag1000G Phase 2 AR1 analysis

This module provides an independently reproducible analysis of the open-access
Phase 2 AR1 release.
The released resource contains 1,142 wild mosquitoes from 13 countries and 234
individuals from 11 laboratory crosses. The atlas uses the released statistically
phased AgamP4 haplotypes and the manuscript's frozen high-\(N_e\) `fastrho`
checkpoint.

The primary atlas contains nine established species-by-population cohorts with at
least 40 diploid mosquitoes: four *Anopheles coluzzii* cohorts and five
*Anopheles gambiae* cohorts. Exactly 40 samples are selected with NumPy seed
2026 from the 1,058 mosquitoes present in every released chromosome-arm HDF5,
and the resulting panel is used unchanged on all five arms. The Phase 2 populations
labelled GM, GW, and KE are excluded from the primary atlas because the Phase 2
resource paper reports uncertain species status; they remain available for a
future sensitivity analysis but are outside this submission.

Every biological claim in this module is computed from the Phase 2 maps and
released Phase 2 metadata. The 2La, pedigree, pyrho, and resistance analyses are
therefore specific to the populations and filtering information available in this release.

The completed replacement contains 45 checksum-bound maps. All map windows passed
the finite/positive-rate QC. The Phase 2 results are: 2La Pearson (r=0.452),
(P=0.222); released-cross Spearman (r_s=0.671), (P=0.00186), across 32
supported 5-Mb windows; matched-subsample mean pyrho concordance (r_s=0.837);
and a 15-region resistance focal-to-control ratio of 0.705 across nine cohorts.
The figures, 13-page main manuscript, and 18-page SI were visually checked. A
`config.json` is therefore `complete` and `submission_eligible=true`.
