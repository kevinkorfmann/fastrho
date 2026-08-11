#!/usr/bin/env bash
#SBATCH --job-name=arabis-resample-ensemble
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
root="${ARABIS_ROOT}/diagnosis/resamples"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${root}/panels.tsv")
IFS=$'\t' read -r panel _ _ _ _ <<< "${line}"
[[ -n "${panel}" ]]
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_ensemble_maps.py" \
  --input-root "${root}/maps/${panel}" \
  --manifest "${ARABIS_SMALLN_ROOT}/frozen_ensemble.json" \
  --output-dir "${root}/maps/${panel}/ensemble" --members 5 --expected-files 8
