#!/usr/bin/env bash
# Reproduce the released folded composite-LD canine bottleneck specialist.
set -euo pipefail
FASTRHO_REPO=${FASTRHO_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to a new output directory}
FASTRHO_PYTHON=${FASTRHO_PYTHON:-python3}
CUDA_DEVICE=${CUDA_DEVICE:-0}
CODE=$FASTRHO_REPO
DP=$MODEL_ROOT
PY=$FASTRHO_PYTHON
export PYTHONNOUSERSITE=1
# Prefer this checkout over any editable installation.
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
cd "$CODE"

NTRAIN=${NTRAIN:-15000}
NTEST=${NTEST:-300}
RADII=${RADII:-5000,25000,50000}
STRIDE=${STRIDE:-0}
MAXNB=${MAXNB:-200}
EPOCHS=${EPOCHS:-50}
DISJOINT=${DISJOINT:-0}            # 1 => disjoint distance bands (cleaner long-range, noisier short)
FORCE_GEN=${FORCE_GEN:-0}         # 1 => regenerate sims even if present

mkdir -p "$DP"
rm -f "$DP/DOGBN_DONE"

if [ "$FORCE_GEN" = "1" ] || [ ! -e "$DP/train_sims/ts_00000000.trees" ]; then
  # dog_gen.dump() skips existing ts files, so a true regen must clear them first
  rm -rf "$DP/train_sims" "$DP/test_sims"
  $PY scripts/dog_gen.py "$DP/train_sims" "$NTRAIN" 0       > "$DP/gen_train.log" 2>&1
  $PY scripts/dog_gen.py "$DP/test_sims"  "$NTEST"  1000000 > "$DP/gen_test.log"  2>&1
  echo "generated $(ls "$DP/train_sims"/*.trees | wc -l) train + $(ls "$DP/test_sims"/*.trees | wc -l) test sims"
else
  echo "reusing existing sims ($(ls "$DP/train_sims"/*.trees | wc -l) train)"
fi

DJ=""; [ "$DISJOINT" = "1" ] && DJ="--disjoint-bands"
rm -rf "$DP/shards"
PREP="--gt-fold --radii $RADII $DJ --stride-after $STRIDE --max-neighbors $MAXNB --num-processes 40"
$PY -m fastrho.preprocess --sim-dir "$DP/train_sims" --out-dir "$DP/shards/train" $PREP > "$DP/prep_train.log" 2>&1
$PY -m fastrho.preprocess --sim-dir "$DP/test_sims"  --out-dir "$DP/shards/test"  $PREP > "$DP/prep_test.log"  2>&1

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" $PY -m fastrho.train --model base --dataset-path "$DP/shards" --gpus 0 \
    --epochs "$EPOCHS" --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$DP/train" \
    --radii "$RADII" $DJ --stride-after "$STRIDE" --max-neighbors "$MAXNB" \
    > "$DP/train.log" 2>&1

ls -t "$DP"/train/fastrho/version_*/checkpoints/*.ckpt | head -1 > "$DP/ckpt.txt"
if [ "${RUN_EMPIRICAL_EVAL:-0}" = "1" ]; then
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" $PY scripts/realdata_infer.py dog dogbn > "$DP/eval_dog.log" 2>&1
fi
touch "$DP/DOGBN_DONE"
echo "DOGBN_DONE"
