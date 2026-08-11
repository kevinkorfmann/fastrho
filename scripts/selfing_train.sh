#!/usr/bin/env bash
# Reproduce the released selfing-aware model.
set -euo pipefail
FASTRHO_REPO=${FASTRHO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to a new output directory}
FASTRHO_PYTHON=${FASTRHO_PYTHON:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-0}
NTRAIN=${NTRAIN:-4000}
NVALID=${NVALID:-300}
SP=$MODEL_ROOT
PY=$FASTRHO_PYTHON
cd "$FASTRHO_REPO"
export PYTHONNOUSERSITE=1
mkdir -p "$SP"
rm -f "$SP/SELF_DONE"
$PY scripts/selfing_gen.py "$SP/train_sims" "$NTRAIN" 0       > "$SP/gen_train.log" 2>&1
$PY scripts/selfing_gen.py "$SP/test_sims"  "$NVALID" 1000000 > "$SP/gen_test.log"  2>&1
$PY -m fastrho.preprocess --sim-dir "$SP/train_sims" --out-dir "$SP/shards/train" --with-features --num-processes 40 > "$SP/prep_train.log" 2>&1
$PY -m fastrho.preprocess --sim-dir "$SP/test_sims"  --out-dir "$SP/shards/test"  --with-features --num-processes 40 > "$SP/prep_test.log"  2>&1
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" $PY -m fastrho.train --model base --dataset-path "$SP/shards" --epochs 50 --gpus 0 \
    --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$SP/train" > "$SP/train.log" 2>&1
ls -t "$SP"/train/fastrho/version_*/checkpoints/*.ckpt | head -1 > "$SP/ckpt.txt"
if [ "${RUN_EMPIRICAL_EVAL:-0}" = "1" ]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" $PY scripts/realdata_infer.py athal self > "$SP/eval_athal.log" 2>&1
fi
touch "$SP/SELF_DONE"
echo "SELF_DONE"
