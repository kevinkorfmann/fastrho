<h1 align="center">fastrho</h1>

<p align="center">
  <strong>Fine-scale recombination maps from population genotypes—without per-dataset training.</strong>
</p>

<p align="center">
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml"><img alt="API tests" src="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml/badge.svg"></a>
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/paper-numbers.yml"><img alt="Paper reproducibility" src="https://github.com/kevinkorfmann/fastrho/actions/workflows/paper-numbers.yml/badge.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-2ea44f.svg"></a>
  <img alt="Release status: research alpha" src="https://img.shields.io/badge/status-research%20alpha-f59e0b">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/kevinkorfmann/fastrho/main/docs/_static_public/fastrho_schematic.png" width="960" alt="fastrho method schematic showing SNP feature tokens, bidirectional state-space inference, and interval-level recombination-rate output">
</p>

`fastrho` is a research package for amortized, fine-scale recombination-map inference. A
bidirectional Mamba2 encoder-decoder reads one order-invariant feature token per SNP and estimates
the population-scaled recombination rate for every adjacent SNP interval.

## At a glance

| | |
|---|---|
| **Input** | Phased or compatible unphased population genotypes, including single-contig VCFs |
| **Output** | Dense adjacent-SNP interval estimates and optional 0-based BED maps |
| **Inference** | One pretrained model with overlapping context inference and no per-dataset retraining |
| **Backbone** | Bidirectional Mamba2 state-space encoder-decoder |
| **Scientific contract** | Exact simulation targets, coordinate checks, provenance ledgers, and paper-number tests |
| **Current stage** | Alpha source release; the primary paper checkpoint is public and checksummed |
| **Active empirical release** | Open Ag1000G Phase 2 AR1: 9 populations × 5 chromosome arms, with 2La, resistance, pedigree, and pyrho result tables |
| **Active comparative SI** | A fixed, provenance-audited panel of 10 species |

The model uses amortized training across diverse simulated demographic histories, enabling
compatible new cohorts to be analyzed in one inference call. Population-scaled rates are reported
directly; absolute-rate point estimates and intervals are conditional on either the supplied
effective population size or the model's auxiliary point estimate, recorded as `Ne_used` with the
result. See
[how to interpret the output](docs/interpretation.md) for practical reporting guidance.

## Install

Install the CPU package and common VCF/DataFrame helpers from PyPI:

```bash
python -m pip install "fastrho[io]"
```

For the complete paper workflow and locked source environment, clone the public repository:

```bash
git clone https://github.com/kevinkorfmann/fastrho.git
cd fastrho
uv sync --frozen --extra figures --extra dev
```

Training and inference require a CUDA environment supported by Mamba. The exact manuscript
environment is recorded in [`requirements/cuda121-paper.txt`](requirements/cuda121-paper.txt).

## Reproducible source workflow

The training command creates a deterministic validation split when the shard directory does not
already contain `train/`, `val/`, or `test/` subdirectories.

```bash
fastrho simulate --data-dir sims --num-ts 2000 --sequence-length 1000000
fastrho preprocess --sim-dir sims --out-dir shards --with-features
fastrho train --model base --dataset-path shards --devices 1 --epochs 40 --seed 0
```

For domain-randomized training, create aligned `hap`, `gt`, and `gtf` shard trees with
`--sfs-shape --r2-debias`, then train with `--dr-base`. These feature flags live in this repository;
no alternate checkout is required.

The exact primary paper model has a numbered portable Slurm workflow, explicit seed schedule,
metric-based checkpoint selection, and byte-level release verification under
[`models/domain-randomized-v1/reproduce/`](models/domain-randomized-v1/reproduce/).

## Prediction with an archived model

Checkpoint statistics are authoritative: `fastrho` reconstructs the exact phased, genotype,
folded, or domain-randomized featurizer from metadata and rejects incompatible inputs.
`feat_stats.npz` is the checkpoint's model companion file—not statistics to compute from the
prediction cohort. Download it with the checkpoint, keep the pair together, and do not modify it
for a new species or VCF.

```bash
python3 scripts/fetch_model_release.py --model-id domain-randomized-v1 \
  --output-dir downloaded-models
```

```bash
fastrho predict --vcf cohort.vcf.gz --chrom 2L \
  --checkpoint downloaded-models/domain-randomized-v1/model.ckpt \
  --stats downloaded-models/domain-randomized-v1/feat_stats.npz \
  --input-mode auto --missing drop-site --out map.bed
```

VCF input is restricted to one contig, missing sites are dropped rather than imputed as reference,
and positions are converted to 0-based BED coordinates. Unphased input is detected from genotype
separators and requires a checkpoint that declares a compatible genotype-token view.

## Phase 2 paper and reproducibility

The current manuscript is:

> **Scalable inference of recombination maps across evolutionary contexts**

The mosquito analysis in the current manuscript is exclusively the open Ag1000G Phase 2 AR1
analysis. Its 45 population-by-arm maps and compact result tables can be
[downloaded directly](docs/data.md); the committed BED release is under
[`paper/anopheles_variants/phase2/release/`](paper/anopheles_variants/phase2/release/). The
comparative SI uses the fixed 10-species panel recorded in
[`paper/figdata/transect.json`](paper/figdata/transect.json).

The authoritative article and SI are `main_phase2.tex` and `si_phase2.tex` in
[`kevinkorfmann/fastrho-manuscript-2026-07-21`](https://github.com/kevinkorfmann/fastrho-manuscript-2026-07-21).
This package repository owns the analysis code and generated artifacts, not a second editable copy
of the paper. [`reproduce/manuscript.lock.json`](reproduce/manuscript.lock.json) pins the exact
manuscript commit and files used by the workflow. Superseded Phase 3 material is legacy and is never
an active input.

The complete ordered reproduction hub is [`reproduce/`](reproduce/). From a fresh clone, one command
recomputes generated results, public downloads, all figures, the numeric audit, both PDFs, and every
paper-specific verification gate:

```bash
./reproduce/run.sh
```

The build writes disposable intermediates to `tmp/` and PDFs to `output/pdf/`. Those directories are
ignored because their tracked sources, figures, tables, snapshots, and generator scripts reproduce
them.

## Repository map

| Path | Purpose |
|---|---|
| [`fastrho/`](fastrho/) | Importable package, CLI, inference, simulation, and training code |
| [`reproduce/`](reproduce/) | One ordered command, machine-readable workflow, and live inventory for the complete paper and SI |
| [`docs/`](docs/) | Minimal user guide: quickstart, Python API, dataset preparation, and interpretation |
| [`examples/manuscript_species/`](examples/manuscript_species/) | Download and inference presets for every empirical manuscript species |
| [`paper/manuscript/`](paper/manuscript/) | Generated figures and TeX inputs staged into the authoritative Phase 2 manuscript |
| [`paper/anopheles_variants/phase2/`](paper/anopheles_variants/phase2/) | Active open-data mosquito maps, results, provenance, figures, and release files |
| [`legacy/`](legacy/) | Superseded analyses, inactive outputs, and historical utilities excluded from Phase 2 |
| [`paper/results_snapshot/`](paper/results_snapshot/) | Frozen numerical inputs used by the paper audit |
| [`research/`](research/) | Frozen Phase 2 supporting workflows for Arabis and demography-matched benchmarks |
| [`research/demography_matched/`](research/demography_matched/) | Ordered Slurm workflow and frozen outputs for the paired ReLERNN/pyrho demographic benchmark |
| [`scripts/`](scripts/) | [Ordered manuscript analysis, figure-generation, and release utilities](scripts/README.md) |
| [`tests/`](tests/) | Software contracts and independent manuscript-number checks |

See [`research/README.md`](research/README.md) and [`paper/README.md`](paper/README.md) for the two
largest project-specific layouts.

## Verification

```bash
python -m pytest tests/test_io_api.py tests/test_phase1_target.py \
  tests/test_phase2_features.py tests/test_stitching.py
python -m pytest tests/paper
python scripts/release_check.py
python scripts/audit_public_releases.py  # online: compare registries with live release assets
python -m build
```

The manuscript suite distinguishes snapshot consistency from raw-array rederivation; a passing
snapshot check alone is not described as end-to-end reproducibility. Every external manuscript
dataset is registered in [`paper/data_provenance.yaml`](paper/data_provenance.yaml) with its version,
access route, terms, citation, local derivatives, and producing scripts.

## Documentation

- [Start here](docs/index.md)
- [Quickstart](docs/quickstart.md)
- [Python API](docs/python-api.md)
- [Use fastrho with your dataset](docs/your-data.md)
- [Run every manuscript species example](examples/manuscript_species/README.md)
- [Interpret the output](docs/interpretation.md)
- [Download, verify, or retrain checkpoints](docs/checkpoints.md)
- [Machine-readable model registry](fastrho/model_registry.json)
- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Code is released under the
[MIT License](LICENSE); external datasets and pretrained weights retain their own licenses and
terms.
