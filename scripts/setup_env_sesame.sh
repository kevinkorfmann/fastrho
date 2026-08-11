#!/usr/bin/env bash
# Reproducible fastrho environment on the sesame V100 box.
# The ONLY mamba-ssm combo verified to import AND run Mamba2 fused kernels on
# Tesla V100S (sm_70). See memory: sesame-mamba-env.
#
#   bash scripts/setup_env_sesame.sh
#
# Always run the resulting python with PYTHONNOUSERSITE=1 (a stray torch
# 2.12+cu130 lives in ~/.local and would otherwise shadow the venv torch).
set -euo pipefail

UV=/home/kkor/.local/bin/uv
VENV=/home/kkor/venvs/fastrho
VPY=$VENV/bin/python
export PYTHONNOUSERSITE=1

# clean venv (NOT --system-site-packages: that makes uv pull torch 2.12+cu130)
rm -rf "$VENV"
"$UV" venv --python 3.12 "$VENV"

# torch 2.4.0 + cu121 (ABI=False), matches sesame nvcc 12.1
"$UV" pip install --python "$VPY" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# prebuilt mamba/causal wheels (battle-tested pair for torch2.4); --no-deps so
# they cannot drag in a different torch
"$UV" pip install --python "$VPY" --no-deps \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl" \
  "https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu122torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"

# mamba_ssm.__init__ eagerly imports an LM head -> needs an OLD transformers
"$UV" pip install --python "$VPY" "transformers==4.40.2" huggingface_hub

# science + IO stack (numpy pinned for ABI safety across numba/msprime/cyvcf2)
"$UV" pip install --python "$VPY" \
  "numpy==1.26.4" einops "msprime>=1.3" tskit "stdpopsim>=0.3" cyvcf2 numba \
  lightning scikit-learn pytest ruff tqdm

# editable installs (no deps so they don't perturb the pinned stack)
"$UV" pip install --python "$VPY" --no-deps -e /home/kkor/fastcxt
"$UV" pip install --python "$VPY" --no-deps -e /home/kkor/fastrho

echo "OK. Smoke test:"
PYTHONNOUSERSITE=1 "$VPY" -c "import torch,mamba_ssm,causal_conv1d,fastcxt,fastrho; \
from fastcxt.modules import BiMambaBlock; \
print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('all imports OK')"
