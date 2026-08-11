#!/usr/bin/env bash
#SBATCH --job-name=smalln-eval
#SBATCH --partition=genoa-std-mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_cross_eval.py" \
  --cross-map "${ARABIS_WORK}/map6j_742_2082_gen.csv" \
  --sample-sheet "${ARABIS_REPO}/research/arabis/sample_assignments.tsv" \
  --software-versions "${ARABIS_WORK}/software_versions.tsv" \
  --maps-dir "${ARABIS_SMALLN_ROOT}/arabis_maps/ensemble" \
  --json "${ARABIS_SMALLN_ROOT}/arabis_cross_results.json" \
  --npz "${ARABIS_SMALLN_ROOT}/arabis_cross_windows.npz" --draws 10000 \
  --model-campaign arabis_smalln_self3 \
  --model-training-n-lines 8,12,16,24,25,32,50
