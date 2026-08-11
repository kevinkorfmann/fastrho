#!/usr/bin/env bash
#SBATCH --job-name=struct-freeze
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_smalln_freeze.py" \
  --campaign "${ARABIS_STRUCTURED_ROOT}" --seeds 7 \
  --campaign-name arabis_structured_selfing_v1 \
  --training-panel-sizes 8,12,16,24,25,32,50 \
  --preregistration "${ARABIS_STRUCTURED_ROOT}/preregistration.json" \
  --output "${ARABIS_STRUCTURED_ROOT}/frozen_ensemble.json"
