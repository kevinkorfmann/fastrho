# Checkpoints: download, verify, and retrain

Six versioned checkpoints are public. `domain-randomized-v1` is the general default; the other
releases are single-view or biological-regime specialists. Every checkpoint must stay paired with
the `feat_stats.npz` from the same release. Each row below has a public, checksummed download.

| Model | Use when | Supported `input_mode` | Release |
|---|---|---|---|
| `domain-randomized-v1` | one general model must handle several genotype views | `phased`, `unphased`, `unpolarized` | [download](https://github.com/kevinkorfmann/fastrho-models/releases/tag/domain-randomized-v1) |
| `base-v1` | ordinary-demography, polarized haplotypes | `phased` | [download](https://github.com/kevinkorfmann/fastrho-models/releases/tag/base-v1) |
| `composite-ld-v1` | diploid genotypes without phase or reliable polarization | `unpolarized` | [download](https://github.com/kevinkorfmann/fastrho-models/releases/tag/composite-ld-v1) |
| `dog-bottleneck-v1` | canine cohorts matching the released bottleneck prior | `unpolarized` | [download](https://github.com/kevinkorfmann/fastrho-models/releases/tag/dog-bottleneck-v1) |
| `high-ne-v1` | phased high-diversity, high-$N_e$ cohorts such as *Anopheles* | `phased` | [download](https://github.com/kevinkorfmann/fastrho-models/releases/tag/high-ne-v1) |
| `selfing-v1` | predominantly selfing, near-homozygous panels | `phased` | [download](https://github.com/kevinkorfmann/fastrho-models/releases/tag/selfing-v1) |

These models are alternatives, not an ensemble. Read the model card on the release page before
using a specialist outside the system for which it was qualified.

Frozen Arabis ensembles and the canid structure checkpoint used only for paper analyses are in the
[Phase 2 paper-support release](https://github.com/kevinkorfmann/fastrho-models/releases/tag/paper-phase2-checkpoints-v1).
Their exact filenames and hashes are recorded in
[`reproduce/checkpoints.json`](https://github.com/kevinkorfmann/fastrho/blob/main/reproduce/checkpoints.json);
they are separated from the six releases intended for new inference. The paper-support checkpoints
retain the exact trained `state_dict` and loading metadata but omit optimizer and trainer state.
The registry records both each downloadable file hash and its full original-checkpoint hash.

## Download with verification

From the repository root:

```bash
python3 scripts/fetch_model_release.py \
  --model-id domain-randomized-v1 \
  --output-dir downloaded-models
```

To audit availability without downloading roughly 200 MB per checkpoint, compare every declared
asset name and SHA-256 against GitHub's live release metadata:

```bash
python3 scripts/audit_public_releases.py
```

This online audit covers the archive, checkpoint, and companion statistics for all six user models,
plus every frozen paper-support asset. The normal fetcher still verifies the bytes after download;
the metadata audit is an availability and registry-consistency check, not a substitute for that
local byte check.

Replace the model ID with any row above. The fetcher reads the machine registry, downloads the versioned archive, checks the archive digest,
checks every embedded member, and only then unpacks it. The usable files are:

```text
downloaded-models/domain-randomized-v1/model.ckpt
downloaded-models/domain-randomized-v1/feat_stats.npz
```

(feat-stats-file)=
## What is `feat_stats.npz`?

`feat_stats.npz` is a small **model companion archive** created during training. It is part of the
trained model, just like the checkpoint; it is not a summary of your VCF and it is not an input you
need to tune.

The checkpoint contains the learned network weights. The companion archive tells fastrho how to
construct and scale the quantities around that network:

| Contents | Used for |
|---|---|
| feature means and standard deviations for each supported input view | standardizing SNP-token features exactly as they were standardized during training |
| target and $N_e$ scaling values | converting the network's standardized outputs back to log $\rho$ and $N_e$ |
| feature-construction settings | reconstructing the training-time LD radii, neighborhood size, and feature flags |
| input-view and dimension metadata | selecting phased, unphased, or unpolarized statistics and rejecting incompatible files |

The filename is historical: it contains both numerical scaling arrays and model/featurizer metadata.
It does **not** contain genotypes, a reference recombination map, organism-specific allele
frequencies, or parameters that should be estimated from the cohort being mapped.

:::{important}
For ordinary inference, do not edit, replace, or recompute `feat_stats.npz`. Download it with the
checkpoint, keep the two files together, and pass both paths. Computing new feature statistics from
your VCF would change the model's input scale and invalidate the pretrained checkpoint.
:::

## What should I do with it?

For a new cohort, chromosome, population, or species: **nothing**. Use the unchanged archive from
the same release as `model.ckpt`. The domain-randomized archive already contains separate scaling
arrays for its supported phased, unphased, and unpolarized views; `input_mode` selects the correct
set.

```python
from pathlib import Path
import fastrho

bundle = Path("downloaded-models/domain-randomized-v1")

pred = fastrho.quick_map_from_vcf(
    "cohort.vcf.gz",
    bundle / "model.ckpt",
    bundle / "feat_stats.npz",
    contig="chr1",
    input_mode="auto",
    device="cuda:0",
)
```

Only training or fine-tuning creates a new companion archive. The training command fits scaling
values on the **training shards**, writes a new `feat_stats.npz` (or the internal
`feat_stats_dr.npz` for the domain-randomized workflow), and that new file must stay with the new
checkpoint. If a cohort lies outside a released model's biological training domain, choose or train
a qualified model; do not try to repair the mismatch by rescaling the cohort with a new `.npz`.

| Situation | Action |
|---|---|
| Use the released model on compatible new data | use the released `feat_stats.npz` unchanged |
| Switch among supported input views | change `input_mode`; keep the same released archive |
| Use a different released model | use that model's checkpoint **and** companion archive |
| Retrain or fine-tune | keep the newly generated archive with the resulting checkpoint |
| File is missing or came from another model | stop and fetch/verify the correct model bundle |

## Pass and verify the pair

The download command above verifies the archive and both embedded files. Use the paths it prints or
the two paths shown above; there is no separate setup step for `feat_stats.npz`.

For files obtained another way, verify them directly:

```bash
python3 scripts/verify_model_release.py \
  --model-id domain-randomized-v1 \
  --checkpoint model.ckpt --stats feat_stats.npz
```

## What is exactly reproducible

The published files are identified exactly by byte count and SHA-256. Anyone using those bytes,
the declared input view, and the same inference inputs can identify the model used by the paper.
The deterministic ZIP also contains its model card, manifest, license, and `SHA256SUMS`.

Training reproduction is a different guarantee. The frozen workflow records all known seeds,
simulation priors, feature views, optimization settings, and the checkpoint-selection rule. GPU
training can still differ at the bit level across hardware or CUDA libraries. A retrain should be
compared on the held-out benchmark suite; it should not be relabeled as the paper checkpoint unless
its files match the published hashes exactly.

## Retrain on a Slurm cluster

```bash
export MODEL_ROOT=/path/to/new/fastrho-model-reproduction
export FASTRHO_PYTHON=/path/to/cuda/environment/bin/python
scripts/dr_train.sh
```

This entry point only submits Slurm jobs. Sixty-four CPU simulation tasks, six preprocessing tasks,
one GPU training task, and one selection task are connected by explicit dependencies. A fresh
output directory contains the full job ledger, environment capture, exact metrics CSV, every epoch
checkpoint, the selected-checkpoint JSON record, and checksums. See
[`models/domain-randomized-v1/reproduce/`](https://github.com/kevinkorfmann/fastrho/tree/main/models/domain-randomized-v1/reproduce)
for the specification and numbered scripts.
