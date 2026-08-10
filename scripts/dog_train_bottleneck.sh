#!/bin/bash
# Bottleneck-aware dog model, fully ISOLATED from the existing campaign_dog run.
#   code:  /home/kkor/fastrho_dog   (separate checkout; does NOT touch /home/kkor/fastrho)
#   data:  /home/kkor/fastrho_data/campaign_dog_bottleneck
#   GPU:   1
# Long-range DISJOINT-BAND + STRIDED radii so the GT featurizer can see the ~Mb breed LD.
set -e
CODE=/home/kkor/fastrho_dog
DP=/home/kkor/fastrho_data/campaign_dog_bottleneck
PY=/home/kkor/venvs/fastrho/bin/python
export PYTHONNOUSERSITE=1
# shadow the editable-installed /home/kkor/fastrho so the isolated copy is used
export PYTHONPATH="$CODE:$PYTHONPATH"
cd "$CODE"

NTRAIN=${NTRAIN:-4000}
NTEST=${NTEST:-300}
# fine bands (village fine-scale, as in the proven dogmodel) + long bands (breed/bottleneck reach)
RADII=${RADII:-300,2000,15000,75000,300000,1500000}
STRIDE=${STRIDE:-48}
MAXNB=${MAXNB:-400}
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

CUDA_VISIBLE_DEVICES=1 $PY -m fastrho.train --model base --dataset-path "$DP/shards" --gpus 0 \
    --epochs "$EPOCHS" --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$DP/train" \
    --radii "$RADII" $DJ --stride-after "$STRIDE" --max-neighbors "$MAXNB" \
    > "$DP/train.log" 2>&1

ls -t "$DP"/train/fastrho/version_*/checkpoints/*.ckpt | head -1 > "$DP/ckpt.txt"
CUDA_VISIBLE_DEVICES=1 $PY scripts/realdata_infer.py dog dogbn > "$DP/eval_dog.log" 2>&1
touch "$DP/DOGBN_DONE"
echo "DOGBN_DONE"
