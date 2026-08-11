#!/bin/bash
# Improve the selfing model by REAL-DATA epoch selection (root-cause fix for the 15k regression).
# (1) featurize 5 A. thaliana chroms once (CPU, parallel) while (2) retraining self with ALL epochs
# saved (GPU1); (3) sanity-check the known epoch-36 (expect ~0.11); (4) sweep every new epoch,
# selecting on held-out chr1 and reporting chr2-5. Writes the real-selected checkpoint + IMPROVE_DONE.
set -u
cd /home/kkor/fastrho
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
PY=/home/kkor/venvs/fastrho/bin/python
SP=/home/kkor/fastrho_data/campaign_self
CACHE=$SP/athal_token_cache
LOG=$SP/train_allep
STATS=$SP/shards15k/feat_stats.npz
mkdir -p "$CACHE"
rm -f "$SP/IMPROVE_DONE"
exec >>"$SP/improve_campaign.log" 2>&1
echo "================ START $(date) ================"

echo "--- [1] featurize 5 athal chroms (CPU, parallel) ---"
for c in 1 2 3 4 5; do
  $PY scripts/self_featurize_cache.py $c "$CACHE" > "$SP/featcache_c$c.log" 2>&1 &
done

echo "--- [2] retrain self, ALL epochs saved (GPU1) ---"
CUDA_VISIBLE_DEVICES=1 $PY -m fastrho.train --model base --dataset-path "$SP/shards15k" \
    --epochs 50 --gpus 0 --batch-size 48 --lr 4e-4 --workers 8 \
    --save-top-k -1 --log-dir "$LOG" > "$SP/train_allep.log" 2>&1
echo "  training exit=$?  ($(date))"

wait
echo "--- featcache tails ---"; for c in 1 2 3 4 5; do echo -n "  c$c: "; tail -1 "$SP/featcache_c$c.log"; done
echo "  n_ckpts_saved=$(ls $LOG/fastrho/version_*/checkpoints/*.ckpt 2>/dev/null | wc -l)"

echo "--- [3] sanity: score existing epoch-36 self on cache (expect all5 ~0.11) ---"
CUDA_VISIBLE_DEVICES=1 $PY scripts/self_epoch_select.py \
  "$SP/train15k/fastrho/version_0/checkpoints/epoch=36-val_loss=-0.052.ckpt" \
  "$STATS" --cache-dir "$CACHE" --select 1 --report 2,3,4,5 --topk 1 2>&1 \
  | grep -viE "FutureWarning|Warning|@custom|torch|Lightning|rank_zero|GPU avail|Loading|Restoring"

echo "--- [4] real-data epoch sweep over the all-epoch retrain ---"
CUDA_VISIBLE_DEVICES=1 $PY scripts/self_epoch_select.py \
  "$LOG/fastrho/version_*/checkpoints/*.ckpt" "$STATS" \
  --cache-dir "$CACHE" --select 1 --report 2,3,4,5 --topk 6 \
  --out "$SP/ckpt_realselected.txt" 2>&1 \
  | grep -viE "FutureWarning|Warning|@custom|torch|Lightning|rank_zero|GPU avail|Loading|Restoring"

touch "$SP/IMPROVE_DONE"
echo "================ IMPROVE_DONE $(date) ================"
