# Demography-matched competitor benchmark

This analysis separates demographic misspecification from each competitor's attainable performance. It evaluates two versions of both ReLERNN and pyrho on exactly the same bottleneck and expansion data:

| Method | Fixed arm | Matched arm |
|---|---|---|
| ReLERNN | constant-size training simulations | training simulations generated under the true benchmark history |
| pyrho | constant-size lookup table | lookup table generated under the true benchmark history |

All four arms within a scenario share byte-identical validated VCFs, truth maps, tree sequences, and genome files. Validation is outcome-blind: before any method runs, it removes the small number of rows containing genotype alleles outside the biallelic `0/1` encoding accepted by both workflows. The raw VCFs remain in the frozen archive, and the manifest records raw and validated hashes plus every removed coordinate. The registered primary endpoint is pooled Pearson correlation at 25 kb. The 100-kb correlation, Spearman correlation, median estimated-to-true rate ratio, and number of scored windows are secondary endpoints. Paired metrics use the same jointly valid windows: truth and both predictions must be finite and strictly positive. The complete frozen rules and settings are in `design.json`. A checksum-verified, fixed-model fastrho prediction from the original benchmark is rescored over all 24 byte-identical regions as an auxiliary reference, not a demographic arm; its provenance is recorded in the reference manifest and score files.

## Reproduce on Betty

Run from a Betty login node with a new output directory. `submit.sh` only submits Slurm jobs; every computational stage runs inside an allocation.

```bash
git clone https://github.com/kevinkorfmann/fastrho.git
cd fastrho
export DEMO_ROOT=/vast/projects/smathi/cohort/$USER/fastrho-demography-reproduction
research/demography_matched/slurm/submit.sh
```

The workflow expands the frozen evaluation inputs from `docs/data/downloads/demography_matched_inputs.zip`, creates pinned environments, validates ReLERNN on a Betty B200 GPU, and submits the paired arrays. Use a fresh `DEMO_ROOT`: preparation and ReLERNN simulation deliberately refuse to overwrite an existing run.

## Ordered workflow

| Stage | Script | Purpose |
|---:|---|---|
| 00 | `00_pyrho_environment.sbatch` | Install and verify the archived Python 3.12 pyrho/ldpop stack. |
| 01 | `01_relernn_environment.sbatch` | Pull the pinned Blackwell-compatible NVIDIA image and install ReLERNN commit `6655efd`. |
| 02 | `02_prepare.sbatch` | Expand frozen inputs, validate VCF rows identically for all arms, create paired hard-linked arms, and record input hashes. |
| 03 | `03_relernn_smoke.sbatch` | Require GPU visibility and complete a one-epoch simulate/train/predict smoke test. |
| 04 | `04_relernn_simulate.sbatch` | Generate 100,000/10,000/10,000 train/validation/test examples for each ReLERNN arm. |
| 05 | `05_relernn_train_score.sbatch` | Train for 100 epochs, predict, and score each ReLERNN arm. |
| 06 | `06_pyrho_infer_score.sbatch` | Build the fixed or matched table, infer maps, and score each pyrho arm. |
| 07 | `07_collate.sbatch` | Validate paired window counts and write the final comparison and checksums. |

`submit.sh` encodes the dependency graph and records all job IDs in `submission.tsv`. CPU simulation and table construction run as four-way arrays; ReLERNN training uses four independent B200 jobs.
The training stage excludes `dgx015`, which repeatedly terminated jobs before the submitted script
started during this campaign; no model setting or input was changed in the retries.

## Analysis code and outputs

- `scripts/prepare_demography_benchmark.py` creates the paired input arms and checksum manifest.
- `scripts/run_relernn_paired.py` freezes the ReLERNN simulation, training, and prediction commands.
- `scripts/run_pyrho_config.py` builds the lookup table and runs pyrho per region.
- `scripts/capture_demography_runtime.py` validates and records the exact runtimes.
- `scripts/collate_demography_benchmark.py` rederives and combines the registered metrics on joint paired support.
- `scripts/archive_demography_predictions.py` archives every truth and prediction track on the
  registered 25-kb grid.
- `scripts/score_fastrho_reference.py` rescored the fixed-model reference over all frozen regions.

The `diagnostics/` directory contains only named, one-purpose technical checks. In particular,
`score_fastrho_reference.sbatch` verifies the auxiliary baseline independently before collation;
it does not alter or select any registered arm.

The main result is `${DEMO_ROOT}/results/paired_demography_results.json`. It contains every registered metric and the observed range of ReLERNN's data-dependent native windows. The companion `${DEMO_ROOT}/results/paired_demography_predictions.npz` stores the truth and every arm on the common 25-kb grid for replotting. `${DEMO_ROOT}/input_manifest.json` proves that each pair used identical evaluation data and lists the deterministic VCF validation; `${DEMO_ROOT}/runtime/` records package and container versions; per-arm ReLERNN manifests record commands, seeds, prediction hashes, and training sizes; and `${DEMO_ROOT}/results/SHA256SUMS` binds the design, inputs, runtimes, and final artifacts. The compact, source-controlled retrieval is under `results/`; its README maps those files to the full Betty run.

The ReLERNN setup applies the compatibility patch in `patches/relernn_keras3_hdf5.patch`. The pinned Keras 3 runtime cannot faithfully reload this legacy GRU through whole-model serialization, so the patch saves its trained weights and rebuilds the same fixed architecture before prediction. Simulation, architecture, optimization, evaluation, and prediction are unchanged. The setup job verifies that only the training, prediction, and helper modules differ from commit `6655efd` and records the patch hash.

Technical failures may be repaired only if the frozen data, histories, seeds, hyperparameters, and endpoints remain unchanged. Every completed arm is reported; no run is selected by its effect on Arabis or on a manuscript claim.
