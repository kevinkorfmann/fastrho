<h1 align="center">fastrho</h1>

<p align="center">
  <strong>Fine-scale recombination maps from population genotypes—without per-dataset training.</strong>
</p>

<p align="center">
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml"><img alt="API tests" src="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/paper-numbers.yml"><img alt="Repository verification" src="https://github.com/kevinkorfmann/fastrho/actions/workflows/paper-numbers.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/docs.yml"><img alt="Documentation" src="https://github.com/kevinkorfmann/fastrho/actions/workflows/docs.yml/badge.svg?branch=main&event=push"></a>
  <a href="https://doi.org/10.64898/2026.08.20.746066"><img alt="bioRxiv preprint" src="https://img.shields.io/badge/bioRxiv-preprint-B31B1B"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/License-MIT-2ea44f.svg"></a>
  <img alt="Release status: research alpha" src="https://img.shields.io/badge/status-research%20alpha-f59e0b">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/kevinkorfmann/fastrho/main/docs/_static_public/fastrho_schematic.png" width="960" alt="Paper schematic showing the mapping from population genotypes to interval tokens, bidirectional state-space inference, interval recombination-rate output, and a regional effective-population-size estimate">
</p>

`fastrho` infers fine-scale recombination maps from population genotypes using pretrained neural
sequence models. It accepts phased or compatible unphased single-contig VCFs and writes dense
interval estimates or 0-based BED maps. You do not need to train a model for each dataset.

> **Status:** research alpha. Inference requires an NVIDIA GPU with a CUDA environment compatible
> with PyTorch and Mamba-SSM.

## Quick start

Install the package and VCF helpers:

```bash
python -m pip install "fastrho[io]"
```

Download a verified model bundle. Keep `model.ckpt` and `feat_stats.npz` together; the statistics
belong to the checkpoint and must not be recomputed from your VCF.

```bash
fastrho-fetch-model --model-id domain-randomized-v1 \
  --output-dir downloaded-models
```

Infer one chromosome:

```bash
fastrho predict --vcf cohort.vcf.gz --chrom 2L \
  --checkpoint downloaded-models/domain-randomized-v1/model.ckpt \
  --stats downloaded-models/domain-randomized-v1/feat_stats.npz \
  --input-mode auto --missing drop-site --out map.bed
```

The input must resolve to one contig. Missing sites are dropped rather than filled as reference.
`input-mode auto` detects phased versus unphased separators, but it cannot determine ancestral
allele orientation. See the [quickstart](https://fastrho.readthedocs.io/en/latest/quickstart.html)
before analyzing a new cohort.

## Understand the output

- `rho_per_bp` is the population-scaled rate, $\rho=4N_e r$, per base pair.
- `r_per_bp` is conditional on the supplied or estimated `Ne_used`.
- Multiply `r_per_bp` by $10^8$ to obtain cM/Mb.
- Do not compare absolute rates across populations unless the $N_e$ scaling is appropriate.

Read the [interpretation guide](https://fastrho.readthedocs.io/en/latest/interpretation.html) before
making biological claims.

## Documentation

- [Quickstart](https://fastrho.readthedocs.io/en/latest/quickstart.html)
- [Prepare your dataset](https://fastrho.readthedocs.io/en/latest/your-data.html)
- [Python API](https://fastrho.readthedocs.io/en/latest/python-api.html)
- [Download checkpoints](https://fastrho.readthedocs.io/en/latest/checkpoints.html)
- [Download inferred maps and paper data](https://fastrho.readthedocs.io/en/latest/data.html)

## Paper, data, and development

The authoritative manuscript is in the separate
[`fastrho-manuscript-2026-07-21`](https://github.com/kevinkorfmann/fastrho-manuscript-2026-07-21)
repository. Its active sources are `main.tex` and `si.tex`. This repository provides the analysis
code, checkpoint registry, inferred-map downloads, and provenance records. Historical frozen
reproduction material under [`reproduce/`](reproduce/) is retained for provenance but is not the
authority for the current manuscript or the public repository checks.

Developers can start with [CONTRIBUTING.md](CONTRIBUTING.md). The exact paper environment is recorded
in [`requirements/cuda121-paper.txt`](requirements/cuda121-paper.txt).

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Code is released under the
[MIT License](LICENSE); external datasets and pretrained weights retain their own licenses and
terms.
