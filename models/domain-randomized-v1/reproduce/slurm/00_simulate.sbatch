#!/usr/bin/env bash
#SBATCH --job-name=fastrho-dr-sim
#SBATCH --partition=genoa-std-mem
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=5632M
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/models/domain-randomized-v1/reproduce/slurm/common.sh"

task=${SLURM_ARRAY_TASK_ID}
if (( task < 60 )); then
  split=train
  count=250
  offset=$((task * count))
else
  split=validation
  count=100
  offset=$(((task - 60) * count))
fi

"${FASTRHO_PYTHON}" -m fastrho.simulate \
  --data-dir "${MODEL_ROOT}/simulations/${split}" \
  --num-ts "${count}" --seed-offset "${offset}" \
  --num-processes "${SLURM_CPUS_PER_TASK}" \
  --sequence-length 1000000 --window-size 2000
