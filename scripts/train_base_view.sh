#!/usr/bin/env bash
# Reproduce either the phased base-v1 or folded composite-ld-v1 training design.
set -euo pipefail

MODEL_ID=${1:?usage: scripts/train_base_view.sh base-v1|composite-ld-v1}
case "$MODEL_ID" in
  base-v1)
    VIEW_FLAG=--with-features
    VIEW_KIND=hap
    ;;
  composite-ld-v1)
    VIEW_FLAG=--gt-fold
    VIEW_KIND=gtf
    ;;
  *)
    echo "unsupported model: $MODEL_ID" >&2
    exit 2
    ;;
esac

FASTRHO_REPO=${FASTRHO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to a new output directory}
FASTRHO_PYTHON=${FASTRHO_PYTHON:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-0}
NTRAIN=${NTRAIN:-15000}
NVALID=${NVALID:-400}
PROCESSES=${PROCESSES:-40}

cd "$FASTRHO_REPO"
mkdir -p "$MODEL_ROOT/logs"

"$FASTRHO_PYTHON" -m fastrho.simulate \
  --data-dir "$MODEL_ROOT/simulations/train" --num-ts "$NTRAIN" \
  --num-processes "$PROCESSES" > "$MODEL_ROOT/logs/simulate-train.log" 2>&1
"$FASTRHO_PYTHON" -m fastrho.simulate \
  --data-dir "$MODEL_ROOT/simulations/validation" --num-ts "$NVALID" \
  --seed-offset 1000000 --num-processes "$PROCESSES" \
  > "$MODEL_ROOT/logs/simulate-validation.log" 2>&1

"$FASTRHO_PYTHON" -m fastrho.preprocess \
  --sim-dir "$MODEL_ROOT/simulations/train" --out-dir "$MODEL_ROOT/shards/train" \
  "$VIEW_FLAG" --num-processes "$PROCESSES" > "$MODEL_ROOT/logs/preprocess-train.log" 2>&1
"$FASTRHO_PYTHON" -m fastrho.preprocess \
  --sim-dir "$MODEL_ROOT/simulations/validation" --out-dir "$MODEL_ROOT/shards/test" \
  "$VIEW_FLAG" --num-processes "$PROCESSES" > "$MODEL_ROOT/logs/preprocess-validation.log" 2>&1

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$FASTRHO_PYTHON" -m fastrho.train \
  --model base --dataset-path "$MODEL_ROOT/shards" --epochs 50 --gpus 0 \
  --batch-size 48 --lr 4e-4 --workers 10 --seed 0 --save-top-k -1 \
  --featurizer-kind "$VIEW_KIND" --log-dir "$MODEL_ROOT/training" \
  > "$MODEL_ROOT/logs/train.log" 2>&1

echo "completed $MODEL_ID training under $MODEL_ROOT"
