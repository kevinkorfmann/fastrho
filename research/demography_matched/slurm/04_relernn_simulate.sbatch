#!/usr/bin/env bash
#SBATCH --job-name=demo-rel-sim
#SBATCH --partition=genoa-std-mem
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"

arms=(bottleneck_relernn_constant bottleneck_relernn_matched expansion_relernn_constant expansion_relernn_matched)
arm=${arms[${SLURM_ARRAY_TASK_ID}]}
scenario=${arm%%_*}
args=()
if [[ "${arm}" == *_matched ]]; then
  args+=(--demography-history "${DEMO_REPO}/research/demography_matched/demographies/${scenario}_smcpp.csv")
fi
relernn_run python "${DEMO_REPO}/scripts/run_relernn_paired.py" simulate \
  --config-dir "${DEMO_ROOT}/arms/${arm}" --n-cpu "${SLURM_CPUS_PER_TASK}" \
  --n-train 100000 --n-validation 10000 --n-test 10000 --max-sites 256 --seed 1 \
  "${args[@]}"
