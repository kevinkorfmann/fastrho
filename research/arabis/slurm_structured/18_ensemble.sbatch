#!/usr/bin/env bash
#SBATCH --job-name=struct-ensemble
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_ensemble_maps.py" \
  --input-root "${ARABIS_STRUCTURED_ROOT}/arabis_maps" \
  --manifest "${ARABIS_STRUCTURED_ROOT}/frozen_ensemble.json" \
  --output-dir "${ARABIS_STRUCTURED_ROOT}/arabis_maps/ensemble" --members 7
