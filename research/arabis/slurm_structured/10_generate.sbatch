#!/usr/bin/env bash
#SBATCH --job-name=struct-gen
#SBATCH --partition=genoa-std-mem
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
: "${STRUCTURED_SPLIT:?}" "${STRUCTURED_PER_TASK:?}" "${STRUCTURED_OFFSET_BASE:?}"
task=${SLURM_ARRAY_TASK_ID}
chunk=$(printf '%03d' "${task}")
offset=$((STRUCTURED_OFFSET_BASE + task * STRUCTURED_PER_TASK))
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_structured_selfing_gen.py" \
  --out-dir "${ARABIS_STRUCTURED_ROOT}/sims/${STRUCTURED_SPLIT}/chunk_${chunk}" \
  --count "${STRUCTURED_PER_TASK}" --offset "${offset}" \
  --workers "${SLURM_CPUS_PER_TASK}"
