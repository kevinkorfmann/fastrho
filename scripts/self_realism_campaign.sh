#!/bin/bash
# First BGS+selfing realism model: generate a blended selfer training set (SLiM-BGS on the real
# A. thaliana map + real exons + H18 DFE, blended with coalescent-selfing volume), preprocess,
# train all-epochs, and real-data-select the checkpoint on held-out chr1 (report chr2-5).
# Compare the selected recovery to the incumbent self2 = 0.27. GPU0.
set -u
cd /home/kkor/fastrho
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export SLIM_BIN=/home/kkor/.local/bin/slim
PY=/home/kkor/venvs/fastrho/bin/python
SP=/home/kkor/fastrho_data/${SELF_TAG:-campaign_self_realism}
BTRAIN=$SP/blend_train; BTEST=$SP/blend_test; SHARDS=$SP/shards; LOG=$SP/train
STATS=$SHARDS/feat_stats.npz
CACHE=/home/kkor/fastrho_data/campaign_self/athal_token_cache
TOTAL=${SELF_TOTAL:-6000}; FRAC=${SELF_FRAC:-0.33}
SEL=${SELF_SELECT:-1}; REP=${SELF_REPORT:-2,3,4,5}
# LOCO: SELF_EXCLUDE_CHROM holds that chromosome's REAL map out of the blend (read by the gen
# scripts); select the epoch on in-training chroms and report the held-out one for a clean number.
export SELF_EXCLUDE_CHROM=${SELF_EXCLUDE_CHROM:-}
mkdir -p "$SP"
rm -f "$SP/REALISM_DONE" "$SP/GEN_DONE"
exec >>"$SP/campaign.log" 2>&1
echo "================ START $(date)  total=$TOTAL frac_slim=$FRAC ================"

echo "--- [1] generate blend: $TOTAL train (${FRAC} SLiM) + 300 coalescent test ---"
$PY scripts/selfing_blend_gen.py "$BTRAIN" "$TOTAL" "$FRAC" 40 0
$PY scripts/selfing_blend_gen.py "$BTEST" 300 0.0 40 900000
touch "$SP/GEN_DONE"
echo "  train ts=$(ls $BTRAIN/ts_*.trees 2>/dev/null | wc -l)  test ts=$(ls $BTEST/ts_*.trees 2>/dev/null | wc -l)  ($(date))"

echo "--- [2] preprocess (17-feat) ---"
$PY -m fastrho.preprocess --sim-dir "$BTRAIN" --out-dir "$SHARDS/train" --with-features --num-processes 40
$PY -m fastrho.preprocess --sim-dir "$BTEST"  --out-dir "$SHARDS/test"  --with-features --num-processes 40

echo "--- [3] train, ALL epochs saved (GPU0) ---"
CUDA_VISIBLE_DEVICES=0 $PY -m fastrho.train --model base --dataset-path "$SHARDS" \
    --epochs 50 --gpus 0 --batch-size 48 --lr 4e-4 --workers 8 --save-top-k -1 --log-dir "$LOG"
echo "  n_ckpts=$(ls $LOG/fastrho/version_*/checkpoints/*.ckpt 2>/dev/null | wc -l)  ($(date))"

echo "--- [4] real-data epoch selection (select chr$SEL, report chr$REP; excluded=$SELF_EXCLUDE_CHROM) ---"
CUDA_VISIBLE_DEVICES=0 $PY scripts/self_epoch_select.py \
    "$LOG/fastrho/version_*/checkpoints/*.ckpt" "$STATS" \
    --cache-dir "$CACHE" --select "$SEL" --report "$REP" --topk 8 --out "$SP/ckpt_realselected.txt" 2>&1 \
    | grep -viE "FutureWarning|@custom|Warning|Restoring|Loading|Reading"

touch "$SP/REALISM_DONE"
echo "================ REALISM_DONE $(date) ================"
