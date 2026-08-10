#!/usr/bin/env bash
#SBATCH --job-name=smalln-gen
#SBATCH --partition=genoa-std-mem
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
: "${SMALLN_SPLIT:?}" "${SMALLN_PER_TASK:?}" "${SMALLN_OFFSET_BASE:?}"
task=${SLURM_ARRAY_TASK_ID}
chunk=$(printf '%03d' "${task}")
offset=$((SMALLN_OFFSET_BASE + task * SMALLN_PER_TASK))
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_smalln_selfing_gen.py" \
  --out-dir "${ARABIS_SMALLN_ROOT}/sims/${SMALLN_SPLIT}/chunk_${chunk}" \
  --count "${SMALLN_PER_TASK}" --offset "${offset}" \
  --workers "${SLURM_CPUS_PER_TASK}"
