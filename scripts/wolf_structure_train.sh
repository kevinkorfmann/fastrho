#!/usr/bin/env bash
set -euo pipefail

FASTRHO_REPO=${FASTRHO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to a new output directory}
FASTRHO_PYTHON=${FASTRHO_PYTHON:-python3}
DOG_CHECKPOINT=${DOG_CHECKPOINT:?set DOG_CHECKPOINT to dog-bottleneck-v1/model.ckpt}
CUDA_DEVICE=${CUDA_DEVICE:-0}
CODE=$FASTRHO_REPO
DATA=$MODEL_ROOT
PY=$FASTRHO_PYTHON
BASE_CKPT=$DOG_CHECKPOINT
RADII=5000,25000,50000

export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
cd "$CODE"
mkdir -p "$DATA"

if [ "$(find "$DATA/train_sims" -maxdepth 1 -name 'ts_*.npz' 2>/dev/null | wc -l)" -lt 1200 ]; then
  $PY scripts/wolf_structure_gen.py "$DATA/train_sims" 1200 0 32
fi
if [ "$(find "$DATA/test_sims" -maxdepth 1 -name 'ts_*.npz' 2>/dev/null | wc -l)" -lt 240 ]; then
  $PY scripts/wolf_structure_gen.py "$DATA/test_sims" 240 2000000 32
fi

if ! compgen -G "$DATA/shards/train/*.npz" > /dev/null; then
  $PY -m fastrho.preprocess --sim-dir "$DATA/train_sims" --out-dir "$DATA/shards/train" \
    --gt-fold --radii "$RADII" --num-processes 32
fi
if ! compgen -G "$DATA/shards/test/*.npz" > /dev/null; then
  $PY -m fastrho.preprocess --sim-dir "$DATA/test_sims" --out-dir "$DATA/shards/test" \
    --gt-fold --radii "$RADII" --num-processes 32
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" $PY -m fastrho.train --model base --dataset-path "$DATA/shards" \
  --gpus 0 --epochs 12 --batch-size 32 --lr 1e-4 --workers 8 \
  --checkpoint "$BASE_CKPT" --log-dir "$DATA/train" --radii "$RADII" \
  > "$DATA/train.log" 2>&1

find "$DATA/train" -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2- > "$DATA/ckpt.txt"
echo "wolf structure model: $(cat "$DATA/ckpt.txt")"
