#!/usr/bin/env bash
#SBATCH --job-name=struct-feat
#SBATCH --partition=genoa-std-mem
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
: "${STRUCTURED_SPLIT:?}"
chunk=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
mkdir -p "${ARABIS_STRUCTURED_ROOT}/shards/${STRUCTURED_SPLIT}"
"${ARABIS_ENV}/bin/python" -m fastrho.preprocess \
  --sim-dir "${ARABIS_STRUCTURED_ROOT}/sims/${STRUCTURED_SPLIT}/chunk_${chunk}" \
  --out-dir "${ARABIS_STRUCTURED_ROOT}/shards/${STRUCTURED_SPLIT}" \
  --with-features --num-processes "${SLURM_CPUS_PER_TASK}"
