#!/bin/bash
# Dedicated dog model: unphased composite-LD (folded) + FINER radii + dog prior. GPU1.
cd /home/kkor/fastrho
DP=/home/kkor/fastrho_data/campaign_dog
PY=/home/kkor/venvs/fastrho/bin/python
export PYTHONNOUSERSITE=1
mkdir -p "$DP"
rm -f "$DP/DOG_DONE"
$PY scripts/dog_gen.py "$DP/train_sims" 4000 0       > "$DP/gen_train.log" 2>&1
$PY scripts/dog_gen.py "$DP/test_sims"  300 1000000  > "$DP/gen_test.log"  2>&1
$PY -m fastrho.preprocess --sim-dir "$DP/train_sims" --out-dir "$DP/shards/train" --gt-fold --radii 300,2000,15000 --num-processes 40 > "$DP/prep_train.log" 2>&1
$PY -m fastrho.preprocess --sim-dir "$DP/test_sims"  --out-dir "$DP/shards/test"  --gt-fold --radii 300,2000,15000 --num-processes 40 > "$DP/prep_test.log"  2>&1
CUDA_VISIBLE_DEVICES=1 $PY -m fastrho.train --model base --dataset-path "$DP/shards" --epochs 50 --gpus 0 \
    --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$DP/train" > "$DP/train.log" 2>&1
ls -t "$DP"/train/fastrho/version_*/checkpoints/*.ckpt | head -1 > "$DP/ckpt.txt"
CUDA_VISIBLE_DEVICES=1 $PY scripts/realdata_infer.py dog dogmodel > "$DP/eval_dog.log" 2>&1
touch "$DP/DOG_DONE"
echo "DOG_DONE"
