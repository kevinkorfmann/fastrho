#!/usr/bin/env bash
#SBATCH --job-name=smalln-feat
#SBATCH --partition=genoa-std-mem
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
: "${SMALLN_SPLIT:?}"
chunk=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
mkdir -p "${ARABIS_SMALLN_ROOT}/shards/${SMALLN_SPLIT}"
"${ARABIS_ENV}/bin/python" -m fastrho.preprocess \
  --sim-dir "${ARABIS_SMALLN_ROOT}/sims/${SMALLN_SPLIT}/chunk_${chunk}" \
  --out-dir "${ARABIS_SMALLN_ROOT}/shards/${SMALLN_SPLIT}" \
  --with-features --num-processes "${SLURM_CPUS_PER_TASK}"
