<h1 align="center">fastrho</h1>

<p align="center">
  <strong>Fine-scale recombination maps from population genotypes, without per-dataset training.</strong>
</p>

<p align="center">
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml"><img alt="API tests" src="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml/badge.svg"></a>
  <a href="https://github.com/kevinkorfmann/fastrho/actions/workflows/api-tests.yml"><img alt="31 tests passing" src="https://img.shields.io/badge/tests-31%20passing-brightgreen?logo=pytest&logoColor=white"></a>
  <a href="https://pypi.org/project/fastrho/"><img alt="PyPI" src="https://img.shields.io/pypi/v/fastrho?logo=pypi&logoColor=white"></a>
  <a href="https://fastrho.readthedocs.io/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Read%20the%20Docs-8CA1AF?logo=readthedocs&logoColor=white"></a>
  <img alt="Tested on Python 3.10 and 3.12" src="https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white">
  <img alt="Research alpha" src="https://img.shields.io/badge/status-research%20alpha-f59e0b">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/kevinkorfmann/fastrho/main/docs/_static_public/fastrho_schematic.png" width="960" alt="fastrho method schematic showing SNP feature tokens, bidirectional state-space inference, and interval-level recombination-rate output">
</p>

`fastrho` is a research package for amortized, fine-scale recombination-map inference. A
bidirectional Mamba-2 encoder-decoder reads one order-invariant feature token per SNP and estimates
the population-scaled recombination rate for every adjacent-SNP interval.

## What it does

| | |
|---|---|
| **Input** | Phased or compatible unphased population genotypes, including single-contig VCFs |
| **Output** | Adjacent-SNP interval estimates and optional 0-based, half-open BED maps |
| **Inference** | One pretrained model with overlapping-context inference; no per-dataset retraining |
| **Backbone** | Bidirectional Mamba-2 state-space encoder-decoder |
| **Rates** | Population-scaled rho and absolute rate conditional on the recorded `Ne_used` |
| **Stage** | Public research alpha |

## Installation

Install the CPU package and common VCF/DataFrame helpers from PyPI:

```bash
python -m pip install "fastrho[io]"
```

GPU inference requires Linux, an NVIDIA GPU, and compatible PyTorch, CUDA, and Mamba-SSM builds.
For the fully locked inference environment, clone the public source repository:

```bash
git clone https://github.com/kevinkorfmann/fastrho.git
cd fastrho
uv sync --frozen --extra inference --extra io
source .venv/bin/activate
```

Inference is tested on Python 3.10 and 3.12. `uv` is recommended for inference because the lockfile
also records the extension build requirements. CPU-only VCF inspection, simulation, and data
conversion do not load the model. Record the environment used for a run with its checkpoint IDs.

## Download the verified model bundle

The `domain-randomized-v1` release contains the checkpoint and its required feature-statistics
companion. Keep the two files together; `feat_stats.npz` is part of the trained model and must not
be recomputed from a prediction cohort.

```bash
fastrho-fetch-model \
  --model-id domain-randomized-v1 \
  --output-dir downloaded-models
```

The public, checksummed model artifacts are also available from the
[`domain-randomized-v1` release](https://github.com/kevinkorfmann/fastrho-models/releases/tag/domain-randomized-v1).

## Make a map

```bash
fastrho predict \
  --vcf cohort.vcf.gz \
  --chrom chr1 \
  --checkpoint downloaded-models/domain-randomized-v1/model.ckpt \
  --stats downloaded-models/domain-randomized-v1/feat_stats.npz \
  --mutation-rate 1.5e-8 \
  --ne 10000 \
  --input-mode auto \
  --missing drop-site \
  --window-size 50000 \
  --out chr1.50kb.bed
```

VCF input is restricted to one contig per prediction call. Missing sites are dropped rather than
imputed as reference, and positions are converted to 0-based, half-open BED coordinates.
`input_mode="auto"` distinguishes phased from unphased genotype separators; it cannot establish
ancestral polarization.

## Python API

```python
from pathlib import Path
import fastrho

bundle = Path("downloaded-models/domain-randomized-v1")

pred = fastrho.quick_map_from_vcf(
    "cohort.vcf.gz",
    bundle / "model.ckpt",
    bundle / "feat_stats.npz",
    contig="chr1",
    mutation_rate=1.5e-8,
    Ne=10_000,
    input_mode="auto",
    missing="drop-site",
    device="cuda:0",
)

df = fastrho.to_dataframe(pred, chrom="chr1")
fastrho.write_bed(pred, "chr1.50kb.bed", chrom="chr1", window_size=50_000)
```

## Before interpreting a map

An LD-based estimate is a population recombination map, not a direct observation of contemporary
crossovers. Demographic history, structure, inversions, selection, relatedness, selfing, and gene
conversion can change the signal. For a new cohort, record the checkpoint and statistics hashes,
input view, filtering, mutation rate, effective population size, coordinates, and reporting window;
then evaluate split-sample repeatability, realistic simulations, or an independent map.

## Documentation

The public documentation contains the software guide, method schematic, and synthetic example.

- [Quickstart](https://fastrho.readthedocs.io/en/latest/quickstart.html)
- [Python API](https://fastrho.readthedocs.io/en/latest/python-api.html)
- [Use fastrho with your dataset](https://fastrho.readthedocs.io/en/latest/your-data.html)
- [Interpret the output](https://fastrho.readthedocs.io/en/latest/interpretation.html)
- [Checkpoints](https://fastrho.readthedocs.io/en/latest/checkpoints.html)
- [Known-answer simulation](https://fastrho.readthedocs.io/en/latest/simulation.html)

## Verification

```bash
python -m pytest tests/test_io_api.py tests/test_phase1_target.py \
  tests/test_phase2_features.py tests/test_stitching.py
python scripts/release_check.py
python -m build
```

## Citation and license

Citation metadata are provided in `CITATION.cff`. Code is licensed under the MIT License; external
datasets and pretrained weights retain their own licenses and terms.
