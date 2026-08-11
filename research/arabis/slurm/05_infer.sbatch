#!/usr/bin/env bash
#SBATCH --job-name=arabis-infer
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
[[ -s "${ARABIS_CHECKPOINT}" && -s "${ARABIS_STATS}" ]]
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_infer.py" \
  --workdir "${ARABIS_WORK}" --checkpoint "${ARABIS_CHECKPOINT}" \
  --stats "${ARABIS_STATS}" --device cuda:0
