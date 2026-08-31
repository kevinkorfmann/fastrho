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
