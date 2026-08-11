# Training workflows for paper models

The released checkpoint bytes are downloaded and verified through
[`docs/checkpoints.md`](../docs/checkpoints.md). Retraining is a separate scientific reproduction:
GPU and CUDA differences can change the resulting bytes, so compare a retrain on the frozen
held-out benchmarks rather than relabeling it as the released checkpoint.

All paths below are repository-relative. Set `MODEL_ROOT` to a new output directory and, when
needed, set `FASTRHO_PYTHON` and `CUDA_DEVICE`. None of these workflows requires Betty or Sesame.

| Model in the paper | Simulation and training entry point |
|---|---|
| `base-v1` | `scripts/train_base_view.sh base-v1` |
| `domain-randomized-v1` | `models/domain-randomized-v1/reproduce/submit.sh` |
| `composite-ld-v1` | `scripts/train_base_view.sh composite-ld-v1` |
| `high-ne-v1` | `scripts/retrain_highne.sh` |
| `selfing-v1` | `scripts/selfing_train.sh` |
| `dog-bottleneck-v1` | `scripts/dog_train_bottleneck.sh` |
| `arabis-smalln-ensemble` | `research/arabis/slurm_smalln/submit.sh` |
| `arabis-structured-ensemble` | `research/arabis/slurm_structured/submit.sh` |
| `canid-structure-paper-analysis` | `scripts/wolf_structure_train.sh` |

The corresponding generators are `fastrho.simulate`, `scripts/selfing_gen.py`,
`scripts/dog_gen.py`, `scripts/arabis_smalln_selfing_gen.py`,
`scripts/arabis_structured_selfing_gen.py`, and `scripts/wolf_structure_gen.py`. The common
preprocessor and trainer are `fastrho.preprocess` and `fastrho.train`. Model manifests record the
released region counts, selected epochs, input views, and checkpoint hashes; the Phase 2
paper-support ensembles are recorded in `reproduce/checkpoints.json`.
