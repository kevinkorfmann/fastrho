#!/usr/bin/env bash
#SBATCH --job-name=struct-cleantrain
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
for seed in $(seq 0 6); do
  root="${ARABIS_STRUCTURED_ROOT}/seeds/seed${seed}"
  rm -rf "${root}/logs"
  rm -f "${root}/shards/feat_stats.npz"
  mkdir -p "${root}/logs"
done
printf 'cleanup_reason\thardware_only_full_B200_resubmission_before_checkpoint_selection\n' \
  > "${ARABIS_STRUCTURED_ROOT}/training_cleanup.tsv"
