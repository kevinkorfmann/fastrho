#!/bin/bash
#SBATCH --job-name=canid33-infer
#SBATCH --partition=b200-mig45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/infer-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/infer-%j.err

set -euo pipefail
source /etc/profile.d/Z98-lmod.sh 2>/dev/null || true
module load python/3.11.11 cuda/12.8.1

PROJECT=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33
CODE=$PROJECT/repo
PY=/vast/projects/smathi/cohort/korfmann/fastrho-arabis/python/bin/python
export FASTRHO_REALDATA_ROOT=$PROJECT
export STDPOPSIM_CACHE=$PROJECT/cache/stdpopsim
export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=$PROJECT/cache/triton
export NUMBA_CACHE_DIR=$PROJECT/cache/numba
mkdir -p "$PROJECT"/{maps,logs} "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR"

echo "started $(date -Is) on $(hostname)"
nvidia-smi -L
"$PY" -c 'import numpy, scipy, stdpopsim, torch, mamba_ssm; print(numpy.__version__, scipy.__version__, stdpopsim.__version__, torch.__version__)'
"$PY" "$CODE/scripts/wolf_subset_infer.py" \
  wolf33_full_reference "$PROJECT/inputs/dogbn.ckpt" \
  "$PROJECT/inputs/dogbn-feat-stats.npz" \
  "$PROJECT/maps/wolf33_full_reference_dogbn.npz" cuda:0
"$PY" "$CODE/scripts/canid_empirical_bootstrap.py" \
  --wolf "$PROJECT/maps/wolf33_full_reference_dogbn.npz" \
  --dog all_dogs "$PROJECT/maps/dog67_full_reference_dogbn.npz" \
  --dog village_dogs "$PROJECT/maps/dog_village42_full_reference_dogbn.npz" \
  --output "$PROJECT/maps/wolf33_full_reference_bootstrap.json"
echo "finished $(date -Is)"
