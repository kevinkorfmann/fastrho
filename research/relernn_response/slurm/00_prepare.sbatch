#!/usr/bin/env bash
#SBATCH --job-name=relresp-prepare
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/relernn_response/slurm/common.sh"
relernn_run python "${RESPONSE_REPO}/scripts/prepare_relernn_response.py" \
  --source-root "${RESPONSE_SOURCE_ROOT}" \
  --target-root "${RESPONSE_ROOT}" \
  --design "${RESPONSE_REPO}/research/relernn_response/design.json"
