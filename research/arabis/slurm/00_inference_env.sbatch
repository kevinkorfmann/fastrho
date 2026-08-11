#!/usr/bin/env bash
#SBATCH --job-name=arabis-env
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
"${ARABIS_PIXI}" install --manifest-path "${ARABIS_REPO}/research/arabis/cuda/pixi.toml"
CUDA_NVCC=$("${ARABIS_PIXI}" run --manifest-path \
  "${ARABIS_REPO}/research/arabis/cuda/pixi.toml" which nvcc)
export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "${CUDA_NVCC}")")
export PATH="${CUDA_HOME}/bin:${PATH}"
export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TORCH_CUDA_ARCH_LIST=10.0
export MAX_JOBS=${SLURM_CPUS_PER_TASK}
nvcc --version
"${HOME}/.local/bin/uv" sync --frozen --extra train
"${ARABIS_ENV}/bin/python" - <<'PY'
import os

import torch
import mamba_ssm
from fastrho.translate import load_model

assert torch.cuda.is_available(), "Betty B200 CUDA device is unavailable"
model, config, stats = load_model(
    os.environ["ARABIS_CHECKPOINT"], os.environ["ARABIS_STATS"], "cuda:0"
)
del model, stats
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"mamba_ssm={mamba_ssm.__version__}")
print(f"device={torch.cuda.get_device_name(0)}")
print(f"checkpoint={os.environ['ARABIS_CHECKPOINT']}")
print(f"model_config={config}")
PY
