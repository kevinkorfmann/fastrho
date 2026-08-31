#!/usr/bin/env bash
#SBATCH --job-name=relresp-train
#SBATCH --partition=dgx-b200
#SBATCH --exclude=dgx015
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-relernn-response-20260831/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/relernn_response/slurm/common.sh"
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1

arms=(
  bottleneck_constant_prior1_max256
  bottleneck_matched_prior1_max256
  bottleneck_matched_prior1_max1750
)
arm=${arms[${SLURM_ARRAY_TASK_ID}]}
relernn_run python "${RESPONSE_REPO}/scripts/run_relernn_paired.py" train-predict \
  --config-dir "${RESPONSE_ROOT}/arms/${arm}" \
  --train-cpu 8 --epochs 100 --seed 1
