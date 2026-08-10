#!/usr/bin/env bash
#SBATCH --job-name=struct-train
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=28
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
seed_index=${SLURM_ARRAY_TASK_ID}
root="${ARABIS_STRUCTURED_ROOT}/seeds/seed${seed_index}"
"${ARABIS_ENV}/bin/python" -m fastrho.train \
  --model base --dataset-path "${root}/shards" \
  --accelerator gpu --devices 1 --precision 16-mixed \
  --seed "$((41001 + seed_index))" --epochs 40 --lr 4e-4 \
  --batch-size 48 --grad-accum 2 --workers 12 \
  --save-top-k 3 --log-dir "${root}/logs"
