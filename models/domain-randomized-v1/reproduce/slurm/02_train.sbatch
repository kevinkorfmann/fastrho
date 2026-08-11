#!/usr/bin/env bash
#SBATCH --job-name=fastrho-dr-train
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/models/domain-randomized-v1/reproduce/slurm/common.sh"

env_dir="${MODEL_ROOT}/provenance"
mkdir -p "${env_dir}"
"${FASTRHO_PYTHON}" -VV > "${env_dir}/python.txt" 2>&1
"${FASTRHO_PYTHON}" -m pip freeze --all > "${env_dir}/python-packages.txt"
git -C "${FASTRHO_REPO}" rev-parse HEAD > "${env_dir}/repository-commit.txt"
git -C "${FASTRHO_REPO}" status --porcelain=v1 > "${env_dir}/repository-status.txt"
nvidia-smi -q > "${env_dir}/nvidia-smi.txt"

"${FASTRHO_PYTHON}" -m fastrho.train \
  --model base --dr-base "${MODEL_ROOT}/shards" \
  --epochs 55 --accelerator gpu --devices 1 --precision 16-mixed \
  --batch-size 48 --grad-accum 2 --lr 4e-4 --workers 12 --seed 0 \
  --save-top-k -1 --log-dir "${MODEL_ROOT}/training"
