#!/usr/bin/env bash
#SBATCH --job-name=smalln-layout
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
train_count=$(find "${ARABIS_SMALLN_ROOT}/shards/train" -maxdepth 1 -name 'ts_*.npz' | wc -l)
val_count=$(find "${ARABIS_SMALLN_ROOT}/shards/val" -maxdepth 1 -name 'ts_*.npz' | wc -l)
[[ "${train_count}" -eq 8000 ]] || { echo "train shards=${train_count}, expected 8000" >&2; exit 1; }
[[ "${val_count}" -eq 800 ]] || { echo "val shards=${val_count}, expected 800" >&2; exit 1; }
for seed in $(seq 0 4); do
  root="${ARABIS_SMALLN_ROOT}/seeds/seed${seed}"
  mkdir -p "${root}/shards" "${root}/logs"
  ln -sfn "${ARABIS_SMALLN_ROOT}/shards/train" "${root}/shards/train"
  ln -sfn "${ARABIS_SMALLN_ROOT}/shards/val" "${root}/shards/val"
done
printf 'campaign\ttrain_shards\tval_shards\nsmalln_self3\t%s\t%s\n' \
  "${train_count}" "${val_count}" > "${ARABIS_SMALLN_ROOT}/dataset.tsv"
