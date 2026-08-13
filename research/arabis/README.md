# Arabis F2-cross benchmark

This workflow compares species-separated inferred recombination maps with the published
742-individual interspecific F2 linkage map. It deliberately does not label the
cross map as universal truth: the reference represents one parental pair, the
cross contains strong segregation distortion, and linkage information helped
curate parts of the chromosome assembly.

The public inputs are ENA projects `PRJEB39992` (population WGS), `PRJEB33482`
(reference accession 29), and `PRJEB89863` /
assembly `GCA_976985625.1` (F2 experiment and chromosome-level reference), plus
the linkage-map repository at commit
`10c9092ce9e08c16b0a958d4062932262a8c6bc4`.

The F2 offspring are one interspecific mapping data set; they are not split into
parental-species groups. Population WGS is split into the 12-accession
*A. nemorensis* and 25-accession *A. sagittata* panels before inference, and the
two resulting inferred recombination maps are each compared with the same F2 linkage
map. Population reads and linkage-marker physical positions share coordinates
on the chromosome-level *A. nemorensis* assembly `GCA_976985625.1`.

`sample_assignments.tsv` is a direct transcription of Dittberner et al.
Supplementary Table S2. It is authoritative for the 12 *A. nemorensis* and 25
*A. sagittata* accessions because ENA assigns accession 10 to the wrong species.
The source workbook `Supplemental_Tables_final.xlsx` has SHA-256
`c7378e6c64f68dd5b3a850a825492b964915e958a4126fabc9b3f269b2623883`; the
transcribed TSV has SHA-256
`abe0f5dd120e447dbc9e326e044b0a7cad63d549bfb31bffb7da050809ae0f1d`.
The pipeline checks these biological-sample counts before downloading or
calling variants. Accession 29 has three paired-end libraries; it selects
ERR3454147 (5.93 Gb), whose yield is closest to the 6.91-Gb PRJEB39992 ingroup
median, rather than pooling libraries totaling 39.81 Gb before a fixed depth
filter.

On Betty, every executable stage is a Slurm job. Place the frozen checkpoint
and feature statistics in `${ARABIS_ROOT}/model` and submit the complete,
dependency-ordered workflow from the repository root:

```bash
export PATH=/cm/local/apps/slurm/24.11/bin:$PATH
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
bash research/arabis/slurm/submit.sh
```

`submit.sh` schedules preparation, a 37-task resumable ENA download array (20
concurrent tasks), a fully concurrent 37-task alignment array (444 requested
CPU cores), 54 nonoverlapping 5-Mb variant-calling shards (108 requested CPU
cores) across all eight chromosomes, conservative species filtering,
frozen-model GPU inference, and the 1-, 2-, and 5-Mb evaluation. Jobs are linked
with `afterok`; none of the stage scripts will run outside a Slurm allocation.
The pinned linkage map is downloaded at commit
`10c9092ce9e08c16b0a958d4062932262a8c6bc4` and checked against SHA-256
`b980746fe7ffc1b5bbb7691d131bdd8329f8e427995abd2786185383c5273bed`.
After the final job, copy `arabis_cross_results.json` and
`arabis_cross_windows.npz` from `${ARABIS_ROOT}/data` into the corresponding
paper snapshot and figure-data paths, then render the SI figure with
`scripts/fig_arabis_cross.py`.

The structured-map window verification is a separate post-rendering technical check.
It reads no F2 rates and applies the same genome-wide WGS-support and
seven-model-stability criteria to every 2-Mb window. Submit it from the Betty
repository root with `sbatch research/arabis/slurm_structured/21_window_diagnostics.sbatch`;
its frozen output is `paper/results_snapshot/arabis_window_diagnostics.json`.

The primary score is the pooled 2-Mb Spearman correlation after dividing each
map by its chromosome mean. Frozen sensitivities use 1- and 5-Mb windows,
exclude segregation-distorted chromosomes 4 and 7, leave out each chromosome in
turn, remove F2 parents 10 and 69 from their population panels, resample
chromosomes for uncertainty, and circularly shift each inferred
chromosome for a spatial null. The two species are never pooled; the displayed
consensus is the geometric mean of their separately normalized maps.

This is intentionally a zero-shot transfer test. The frozen selfing checkpoint
was trained with 50, 80, 120, 156, or 200 lines per simulation; the 12- and
25-accession Arabis panels (and their 11/24 parent-excluded variants) are below
that envelope. The benchmark therefore measures external robustness at a hard
small-sample boundary and is not presented as in-regime calibration.
