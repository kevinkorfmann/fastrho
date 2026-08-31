#!/usr/bin/env bash
#SBATCH --job-name=relresp-sim
#SBATCH --partition=genoa-std-mem
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/relernn_response/slurm/common.sh"

arms=(
  bottleneck_constant_prior1_max256
  bottleneck_matched_prior1_max256
  bottleneck_matched_prior1_max1750
)
arm=${arms[${SLURM_ARRAY_TASK_ID}]}
max_sites=256
history=()
if [[ "${arm}" == *_max1750 ]]; then
  max_sites=1750
fi
if [[ "${arm}" == bottleneck_matched_* ]]; then
  history+=(--demography-history "${RESPONSE_REPO}/research/demography_matched/demographies/bottleneck_smcpp.csv")
fi
relernn_run python "${RESPONSE_REPO}/scripts/run_relernn_paired.py" simulate \
  --config-dir "${RESPONSE_ROOT}/arms/${arm}" \
  --n-cpu "${SLURM_CPUS_PER_TASK}" \
  --n-train 100000 --n-validation 10000 --n-test 10000 \
  --max-sites "${max_sites}" --upper-rho-theta-ratio 1 --seed 1 \
  "${history[@]}"
