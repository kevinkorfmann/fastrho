#!/usr/bin/env bash
#SBATCH --job-name=relresp-score
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/relernn_response/slurm/common.sh"
relernn_run python "${RESPONSE_REPO}/scripts/score_relernn_native.py" \
  --arm bottleneck_constant_prior16_max256 "${RESPONSE_SOURCE_ROOT}/arms/bottleneck_relernn_constant" \
  --arm bottleneck_matched_prior16_max256 "${RESPONSE_SOURCE_ROOT}/arms/bottleneck_relernn_matched" \
  --arm bottleneck_constant_prior1_max256 "${RESPONSE_ROOT}/arms/bottleneck_constant_prior1_max256" \
  --arm bottleneck_matched_prior1_max256 "${RESPONSE_ROOT}/arms/bottleneck_matched_prior1_max256" \
  --arm bottleneck_matched_prior1_max1750 "${RESPONSE_ROOT}/arms/bottleneck_matched_prior1_max1750" \
  --output "${RESPONSE_ROOT}/results/native_sensitivity.json"
sha256sum "${RESPONSE_ROOT}/design.json" "${RESPONSE_ROOT}/input_manifest.json" \
  "${RESPONSE_ROOT}/results/native_sensitivity.json" \
  > "${RESPONSE_ROOT}/results/SHA256SUMS"
