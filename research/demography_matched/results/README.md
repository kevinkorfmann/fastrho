# Retrieved demography benchmark artifacts

This directory is the compact, checksum-verified record retrieved from the completed Betty campaign. It excludes trained models, environments, logs, and the large raw prediction archive; those are reproducible through the Slurm workflow one directory above.

- `design.json` freezes the paired design and endpoints.
- `input_manifest.json` records the shared evaluation inputs and validation hashes.
- `runtime/` records the pyrho and ReLERNN software environments.
- `manifests/` records each ReLERNN arm's seeds, commands, training sizes, and prediction hash.
- `fastrho_reference_manifest.json` identifies the fixed auxiliary fastrho predictions.
- `fastrho_reference_scores.json` contains their independently rederived scores.
- `paired_demography_results.json` is the final paired result.
- `submission.tsv` records the original Slurm dependency graph.
- `SHA256SUMS` is the checksum ledger written on Betty; paths in that file are relative to the complete cluster result tree.
- `technical_resubmission.json` records the outcome-blind collation repair and replacement Slurm job.

The plotting archive used for unconditional numerical rederivation is stored with the paper figure data. Every reported paired metric is recalculated in tests from that archive, using the same finite, strictly-positive truth and prediction windows in both demographic arms.
