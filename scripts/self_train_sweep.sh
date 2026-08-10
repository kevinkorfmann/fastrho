#!/bin/bash
# Clean retrain (all epochs saved) + real-data epoch sweep for the selfing model.
# Assumes scripts/self_featurize_cache.py is (or is being) run into $CACHE. Uses a fresh log dir
# (train_allep2) so it cannot race the earlier orchestrator. Writes ckpt_realselected.txt + SWEEP_DONE.
set -u
cd /home/kkor/fastrho
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/kkor/venvs/fastrho/bin/python
SP=/home/kkor/fastrho_data/campaign_self
CACHE=$SP/athal_token_cache
LOG=$SP/train_allep2
STATS=$SP/shards15k/feat_stats.npz
rm -f "$SP/SWEEP_DONE"
exec >>"$SP/train_sweep.log" 2>&1
echo "================ START $(date) ================"

echo "--- [1] retrain self, ALL epochs saved (GPU1) ---"
CUDA_VISIBLE_DEVICES=1 $PY -m fastrho.train --model base --dataset-path "$SP/shards15k" \
    --epochs 50 --gpus 0 --batch-size 48 --lr 4e-4 --workers 8 \
    --save-top-k=-1 --log-dir "$LOG" > "$SP/train_allep2.log" 2>&1
echo "  training exit=$?  n_ckpts=$(ls $LOG/fastrho/version_*/checkpoints/*.ckpt 2>/dev/null | wc -l)  ($(date))"

echo "--- [2] wait for the 5 token caches (featurization) ---"
for i in $(seq 1 120); do
  n=$(ls "$CACHE"/athal_c*_tokens.npz 2>/dev/null | wc -l)
  [ "$n" -ge 5 ] && { echo "  cache ready ($n/5)"; break; }
  sleep 30
done
ls -la "$CACHE"/ 2>/dev/null | grep npz

echo "--- [3] sanity: existing epoch-36 self on cache (expect all5 ~0.11) ---"
CUDA_VISIBLE_DEVICES=1 $PY scripts/self_epoch_select.py \
  "$SP/train15k/fastrho/version_0/checkpoints/epoch=36-val_loss=-0.052.ckpt" \
  "$STATS" --cache-dir "$CACHE" --select 1 --report 2,3,4,5 --topk 1 2>&1 \
  | grep -viE "FutureWarning|Warning|@custom|torch|Lightning|rank_zero|GPU avail|Loading|Restoring|Missing|Unexpected"

echo "--- [4] real-data epoch sweep over the all-epoch retrain ---"
CUDA_VISIBLE_DEVICES=1 $PY scripts/self_epoch_select.py \
  "$LOG/fastrho/version_*/checkpoints/*.ckpt" "$STATS" \
  --cache-dir "$CACHE" --select 1 --report 2,3,4,5 --topk 6 \
  --out "$SP/ckpt_realselected.txt" 2>&1 \
  | grep -viE "FutureWarning|Warning|@custom|torch|Lightning|rank_zero|GPU avail|Loading|Restoring|Missing|Unexpected"

touch "$SP/SWEEP_DONE"
echo "================ SWEEP_DONE $(date) ================"
