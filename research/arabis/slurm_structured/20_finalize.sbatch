#!/usr/bin/env bash
#SBATCH --job-name=struct-finalize
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_campaign_finalize.py" \
  --campaign "${ARABIS_STRUCTURED_ROOT}" \
  --output "${ARABIS_STRUCTURED_ROOT}/completion_manifest.json"
