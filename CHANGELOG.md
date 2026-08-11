# Changelog

## Unreleased

- Added a deterministic, self-documented mosquito-map ZIP and explicit AgamP4 coordinate,
  population-scale, absolute-scale, and effective-population-size rescaling guidance.
- Replaced the misleading cohort-mean mosquito `Ne_estimate` export with the exact arm-specific
  `Ne_used` that reproduces every released absolute-rate row.
- Added row-level mosquito scale contracts and an online audit of all public checkpoint assets.
- Synchronized the rendered documentation version with package version 0.1.1.

## 0.1.1 - 2026-08-02

- Added an installed, checksumming `fastrho-fetch-model` command.
- Documented installation from PyPI.
- Published and registered verified base, composite-LD, high-$N_e$, selfing, and dog-bottleneck
  checkpoint bundles with model cards and explicit input contracts.

## 0.1.0 - 2026-07-20

- Removed an obsolete runtime dependency.
- Unified domain-randomized feature flags in the public featurizers.
- Added checkpoint-driven featurizer reconstruction and compatibility checks.
- Corrected chunk-edge weighting and Gaussian-mixture uncertainty aggregation.
- Prevented padded tokens from entering bidirectional sequence scans.
- Made VCF missingness, phase, contig, and coordinate behavior explicit.
- Added complete-region validation crops and deterministic train/validation splitting.
- Added software and model-release metadata checks.

This is an alpha research release. Public pretrained artifacts remain a release gate.
