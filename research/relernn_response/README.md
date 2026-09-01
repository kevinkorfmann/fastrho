# Focused ReLERNN response experiment

This additive workflow addresses the ReLERNN configuration feedback without changing the
submitted benchmark or rerunning unrelated fastrho and pyrho experiments. The submitted code is
frozen by tag `submitted-preprint-2026-08-31`; this workflow runs from the separate
`revision/relernn-response-2026-08-31` branch.

The experiment reuses the byte-identical, validated bottleneck inputs from the paired demographic
benchmark. Two completed arms provide the historical `maxSites=256`,
`upperRhoThetaRatio=16` reference. Three new arms test ReLERNN's documented default ratio of one,
the generating demographic history, and the documented `maxSites=1750` default. All other settings
are frozen in `design.json`.

Run `research/relernn_response/slurm/submit.sh` on Betty. It creates a new root at
`/vast/projects/smathi/cohort/$USER/fastrho-relernn-response-20260831`, submits all computation to
Slurm, and writes `results/native_sensitivity.json`. No stage overwrites the historical run.

## Final intended-regime benchmark

The reviewer-facing result is the independently validated six-scenario suite described by
`intended_regime_diagnostic_design.json`. The first complete valid corrected training suite was
frozen under `provenance/matched_betty_20260901`; models were never mixed across replicas or
selected by test correlation. Those six models use `maxSites=1750`, phased input, each scenario's
generating mutation rate, and `upperRhoThetaRatio=3` for human-like scenarios or 8 for dog.

The frozen models were reused without retraining on 20 prespecified 10-Mb constant-rate test
regions per scenario. Truth and all three methods were scored within exact complete ReLERNN native
windows. A paired, same-input add-on deliberately uses constant demographic histories for the
bottleneck and expansion. ReLERNN's optional `BSCORRECT` stage was also completed and retained as
a separate robustness result; the main figure uses the documented core `PREDICT` output rather
than selecting between stages by test correlation.

The final artifacts are:

- `results/intended_regime_constant_rate_native_raw.json`
- `results/intended_regime_constant_rate_native_bscorrect.json`
- `results/intended_regime_misspecified_complete_native_raw.json`
- `results/intended_regime_validation_report.json`
- `results/revision_timing_stages.json`
- `paper/manuscript/figures/fig1_method_validation_relernn_revision.pdf`

`scripts/validate_relernn_revision_suite.py` verifies the prespecified settings, source-model and
truth checksums, held-out performance, prior support, complete native windows, missing/nonfinite
predictions, finite boundary-zero predictions, and paired-control input identity.
