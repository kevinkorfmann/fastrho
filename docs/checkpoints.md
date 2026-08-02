# Checkpoints: choose, download, and verify

fastrho publishes one general model and five qualified specialists. Every release contains a
checkpoint and its mandatory `feat_stats.npz` companion; never mix files between releases.

| Model | Use when | Required `input_mode` |
|---|---|---|
| `domain-randomized-v1` | General use; one model across phased, unphased, and unpolarized data | matching view |
| `base-v1` | Ordinary-demography, phased and ancestrally polarized haplotypes | `phased` |
| `composite-ld-v1` | Unphased diploid genotypes without trustworthy ancestral polarization | `unpolarized` |
| `high-ne-v1` | Very large effective population size, especially mosquitoes and other dipterans | `phased` |
| `selfing-v1` | Predominantly selfing, near-homozygous accessions represented as haplotypes | `phased` |
| `dog-bottleneck-v1` | Village-dog/breed-like severe bottleneck histories | `unpolarized` |

Read the model card linked by the registry before choosing a specialist. A biological label does not
override the input contract: notably, `selfing-v1` is a phased haplotype model, while the dog model
uses folded unpolarized genotype tokens.

If an older installed fastrho version reports an unknown specialist ID, install the current source
release or download the bundle directly from its linked GitHub Release. The artifact remains
verifiable from the archive SHA-256 in the registry and release page.

## Download with verification

```bash
fastrho-fetch-model \
  --model-id domain-randomized-v1 \
  --output-dir downloaded-models
```

Change `--model-id` to any available model above. For example, download the mosquito/high-$N_e$
specialist with `--model-id high-ne-v1`.

The fetcher reads the machine registry, downloads the versioned archive, checks the archive digest,
checks every embedded member, and only then unpacks it. For the example above, the usable files are:

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
the same release as `model.ckpt`. The domain-randomized archive contains separate scaling arrays for
its supported views; a single-view specialist contains metadata that selects and enforces its one
training-time featurizer.

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
fastrho-verify-model \
  --model-id domain-randomized-v1 \
  --bundle downloaded-models/domain-randomized-v1
```

## What is exactly reproducible

The published files are identified exactly by byte count and SHA-256. Anyone using those bytes,
the declared input view, and the same inference inputs can identify the exact trained model.
The deterministic ZIP also contains its model card, manifest, license, and `SHA256SUMS`.

Training reproduction is a different guarantee. The frozen workflow records all known seeds,
simulation priors, feature views, optimization settings, and the checkpoint-selection rule. GPU
training can still differ at the bit level across hardware or CUDA libraries. A retrain should be
compared on the held-out benchmark suite; it should not be relabeled as the released checkpoint unless
its files match the published hashes exactly.

## Experimental checkpoints not released as defaults

The retained `rich` large-sample-size checkpoint remains a research artifact. It was developed for
roughly 200-haplotype neutral human panels, but its own evaluation did not establish a general
advantage over pyrho in that regime. It is therefore not in the public registry. Large sample size
is also different from large effective population size: use `high-ne-v1` for the latter.

## Retraining

Retraining is an advanced workflow. A complete end-to-end retraining specification is not yet part
of this alpha release; do not treat a newly trained checkpoint as interchangeable with the released
model unless its files match the published hashes exactly.
