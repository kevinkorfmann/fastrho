#!/usr/bin/env bash
#SBATCH --job-name=fastrho-dr-prep
#SBATCH --partition=genoa-std-mem
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=5632M
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/models/domain-randomized-v1/reproduce/slurm/common.sh"

views=(hap gt gtf)
splits=(train validation)
view=${views[$((SLURM_ARRAY_TASK_ID % 3))]}
split=${splits[$((SLURM_ARRAY_TASK_ID / 3))]}
out_split=train
if [[ "${split}" == validation ]]; then
  out_split=test
fi
case "${view}" in
  hap) view_flag=--with-features ;;
  gt) view_flag=--gt ;;
  gtf) view_flag=--gt-fold ;;
  *) exit 2 ;;
esac

"${FASTRHO_PYTHON}" -m fastrho.preprocess \
  --sim-dir "${MODEL_ROOT}/simulations/${split}" \
  --out-dir "${MODEL_ROOT}/shards/${view}/${out_split}" \
  "${view_flag}" --sfs-shape --r2-debias \
  --num-processes "${SLURM_CPUS_PER_TASK}"
