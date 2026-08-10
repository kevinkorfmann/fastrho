#!/usr/bin/env bash
#SBATCH --job-name=demo-rel-train
#SBATCH --partition=dgx-b200
#SBATCH --exclude=dgx015
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1

arms=(bottleneck_relernn_constant bottleneck_relernn_matched expansion_relernn_constant expansion_relernn_matched)
arm=${arms[${SLURM_ARRAY_TASK_ID}]}
config="${DEMO_ROOT}/arms/${arm}"
relernn_run python "${DEMO_REPO}/scripts/run_relernn_paired.py" train-predict \
  --config-dir "${config}" --train-cpu 8 --epochs 100 --seed 1
prediction=$(relernn_run python - "${config}/relernn_run_manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["prediction"])
PY
)
relernn_run python "${DEMO_REPO}/scripts/bench.py" ingest \
  --config "${config}" --kind relernn --predict "${prediction}"
relernn_run python "${DEMO_REPO}/scripts/bench.py" score \
  --config "${config}" --methods relernn --results "${DEMO_ROOT}/results"
