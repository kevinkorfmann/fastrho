#!/bin/bash
# Train a dedicated phase- AND polarization-invariant model (fastrho-GT, folded composite-LD)
# and evaluate it on the hardest cross-species case. ~50-70 min on one GPU.
set -e
cd /home/kkor/fastrho
export PYTHONNOUSERSITE=1
PY=/home/kkor/venvs/fastrho/bin/python
CAMP=/home/kkor/fastrho_data/campaign
M=$CAMP/markers
SH=$CAMP/shards_gtf
LOG=$CAMP/logs
mkdir -p "$SH" "$LOG" "$M"
rm -f "$M/GTMODEL_DONE"

# 1. re-featurize train + test sims with folded composite-LD tokens
$PY -m fastrho.preprocess --sim-dir "$CAMP/train_sims" --out-dir "$SH/train" --gt-fold --num-processes 40 > "$LOG/prep_gtf_train.log" 2>&1
$PY -m fastrho.preprocess --sim-dir "$CAMP/test_sims"  --out-dir "$SH/test"  --gt-fold --num-processes 40 > "$LOG/prep_gtf_test.log" 2>&1

# 2. train (same recipe as the base model)
$PY -m fastrho.train --model base --dataset-path "$SH" --epochs 50 --gpus 0 \
    --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$CAMP/train_gtf" > "$LOG/train_gtf.log" 2>&1
CKPT=$(ls -t "$CAMP"/train_gtf/fastrho/version_*/checkpoints/*.ckpt | head -1)
echo "$CKPT" > "$CAMP/ckpt_gtf.txt"

# 3. evaluate the dedicated model on unphased + unpolarized (its matched condition)
$PY scripts/stdpopsim_maps.py unphased_unpol_gt --ckpt "$CKPT" \
    --stats "$SH/feat_stats.npz" --tag _gtmodel > "$LOG/eval_gtf.log" 2>&1

touch "$M/GTMODEL_DONE"
echo "GTMODEL_DONE"
