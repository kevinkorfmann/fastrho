#!/bin/bash
# Train a selfing-aware fastrho model and evaluate on Arabidopsis (south Sweden). GPU0.
cd /home/kkor/fastrho
SP=/home/kkor/fastrho_data/campaign_self
PY=/home/kkor/venvs/fastrho/bin/python
export PYTHONNOUSERSITE=1
mkdir -p "$SP"
rm -f "$SP/SELF_DONE"
$PY scripts/selfing_gen.py "$SP/train_sims" 4000 0       > "$SP/gen_train.log" 2>&1
$PY scripts/selfing_gen.py "$SP/test_sims"  300 1000000  > "$SP/gen_test.log"  2>&1
$PY -m fastrho.preprocess --sim-dir "$SP/train_sims" --out-dir "$SP/shards/train" --with-features --num-processes 40 > "$SP/prep_train.log" 2>&1
$PY -m fastrho.preprocess --sim-dir "$SP/test_sims"  --out-dir "$SP/shards/test"  --with-features --num-processes 40 > "$SP/prep_test.log"  2>&1
CUDA_VISIBLE_DEVICES=0 $PY -m fastrho.train --model base --dataset-path "$SP/shards" --epochs 50 --gpus 0 \
    --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$SP/train" > "$SP/train.log" 2>&1
ls -t "$SP"/train/fastrho/version_*/checkpoints/*.ckpt | head -1 > "$SP/ckpt.txt"
CUDA_VISIBLE_DEVICES=0 $PY scripts/realdata_infer.py athal self > "$SP/eval_athal.log" 2>&1
touch "$SP/SELF_DONE"
echo "SELF_DONE"
