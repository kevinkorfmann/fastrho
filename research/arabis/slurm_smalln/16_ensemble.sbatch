#!/usr/bin/env bash
#SBATCH --job-name=smalln-ensemble
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_ensemble_maps.py" \
  --input-root "${ARABIS_SMALLN_ROOT}/arabis_maps" \
  --manifest "${ARABIS_SMALLN_ROOT}/frozen_ensemble.json" \
  --output-dir "${ARABIS_SMALLN_ROOT}/arabis_maps/ensemble" --members 5
