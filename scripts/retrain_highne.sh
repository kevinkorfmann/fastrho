#!/usr/bin/env bash
# Retrain the high-Ne (dipteran) model with the cap-mode fix: at high Ne, SHORTEN the region
# instead of crushing the per-bp rate, so the model finally trains on realistic high rho and
# stops systematically under-predicting the absolute recombination rate (the Drosophila/Ag1000G
# bias). New campaign dir so campaign2 stays intact for comparison.
#
#   CUDA=0 bash scripts/retrain_highne.sh
set -uo pipefail

CAMP=/home/kkor/fastrho_data/campaign_hidip
FVPY=/home/kkor/venvs/fastrho/bin/python
LOG=$CAMP/logs
export PYTHONNOUSERSITE=1
cd /home/kkor/fastrho
mkdir -p "$LOG"
rm -f "$CAMP/RETRAIN_DONE" "$CAMP/RETRAIN_FAIL"
fail() { echo "STAGE FAILED: $1"; touch "$CAMP/RETRAIN_FAIL"; exit 1; }

# broadened high-Ne dipteran prior + the fix:
#   --log10-ne-max 6.3        Ne up to ~2e6 (mosquito/Drosophila)
#   --log10-mean-rec-max -7.3 mean rec up to ~5e-8 (covers real dipteran rates; peaks higher via map shape)
#   --cap-mode shorten        keep realistic per-bp rate, trim L to the floor instead of scaling rate down
#   --sequence-length 30000   campaign2's base region length; shorten trims further only when needed
SIM_FLAGS="--sequence-length 30000 --log10-ne-max 6.3 --log10-mean-rec-max -7.3 --cap-mode shorten --min-sequence-length 2000 --max-local-rho 300 --highne-constant-above 2e5"

echo "=== stage 1: simulate (CPU) ==="
$FVPY -m fastrho.simulate --data-dir "$CAMP/train_sims" --num-ts 6000 --num-processes 40 $SIM_FLAGS \
  > "$LOG/sim_train.log" 2>&1 || fail "sim_train"
$FVPY -m fastrho.simulate --data-dir "$CAMP/test_sims"  --num-ts 400  --num-processes 40 $SIM_FLAGS \
  > "$LOG/sim_test.log" 2>&1 || fail "sim_test"

echo "=== stage 2: preprocess -> shards (CPU) ==="
$FVPY -m fastrho.preprocess --sim-dir "$CAMP/train_sims" --out-dir "$CAMP/shards/train" --with-features --num-processes 40 \
  > "$LOG/prep_train.log" 2>&1 || fail "prep_train"
$FVPY -m fastrho.preprocess --sim-dir "$CAMP/test_sims"  --out-dir "$CAMP/shards/test"  --with-features --num-processes 40 \
  > "$LOG/prep_test.log" 2>&1 || fail "prep_test"

echo "=== stage 3: train (GPU) ==="
CUDA_VISIBLE_DEVICES="${CUDA:-0}" $FVPY -m fastrho.train --model base --dataset-path "$CAMP/shards" \
  --epochs 50 --gpus 0 --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$CAMP/train" \
  > "$LOG/train.log" 2>&1 || fail "train"

CKPT=$(ls -t "$CAMP"/train/fastrho/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -s "$CKPT" ] || fail "no checkpoint produced"
echo "$CKPT" > "$CAMP/ckpt.txt"
echo "=== DONE -> $CKPT ==="
touch "$CAMP/RETRAIN_DONE"
