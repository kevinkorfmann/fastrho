# fastrho

**Fine-scale recombination maps from population genotypes—without training a new model for every dataset.**

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Use fastrho
:link: quickstart
:link-type: doc
:class-card: quick-card landing-card

Install the package, download a verified checkpoint, and turn a cohort VCF into a BED map.

**Start the quickstart →**
:::

:::{grid-item-card} Get the inferred maps
:link: data
:link-type: doc
:class-card: quick-card landing-card

Download the Ag3 mosquito atlas and the paper's *Arabis*, *Arabidopsis*, redpoll, canid, and
cross-species maps.

**Browse map downloads →**
:::

::::

## Start in two commands

```bash
python -m pip install "fastrho[io]"
fastrho-fetch-model --model-id domain-randomized-v1 --output-dir downloaded-models
```

Inference requires a compatible NVIDIA GPU. Keep each checkpoint's `model.ckpt` and
`feat_stats.npz` together; the {doc}`quickstart` shows the complete VCF-to-map command.

:::{figure} _static_public/fastrho_schematic.png
:alt: Paper schematic showing construction of interval tokens from population genotypes and bidirectional state-space inference of a recombination map.
:class: hero-figure
:width: 100%

Population genotypes are converted to interval features and passed through the pretrained
bidirectional sequence model to produce a recombination map with uncertainty estimates.
:::

## More documentation

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Prepare your data
:link: your-data
:link-type: doc
:class-card: quick-card

Choose the input view, cohort, filters, checkpoint, and reporting scale.
:::

:::{grid-item-card} Python API
:link: python-api
:link-type: doc
:class-card: quick-card

Map one chromosome or load a model once for a whole genome.
:::

:::{grid-item-card} Models and checkpoints
:link: checkpoints
:link-type: doc
:class-card: quick-card

Download verified model bundles and check their supported inputs.
:::

:::{grid-item-card} Interpret a map
:link: interpretation
:link-type: doc
:class-card: quick-card

Understand $\rho$, absolute rates, uncertainty, and LD-based limitations.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

quickstart
simulation
python-api
your-data
interpretation
checkpoints
data
```
