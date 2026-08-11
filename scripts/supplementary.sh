#!/usr/bin/env bash
# Supplementary experiments for the preprint, run in the paper phase AFTER the campaign
# model is trained (reuses it; no retraining). Adds: dog real-map recovery (no-PRDM9
# contrast to human), the pyrho wrong-demography panel, and the between-population
# difference + noise-floor experiment. Non-fatal per-step.
export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
FVPY=/home/kkor/venvs/fastrho/bin/python
PVPY=/home/kkor/venvs/pyrho/bin/python
cd /home/kkor/fastrho
CAMP=/home/kkor/fastrho_data/campaign
R=$CAMP/results
CKPT=$(cat "$CAMP/ckpt.txt"); STATS="$CAMP/shards15k/feat_stats.npz"
DEV=cuda:0

echo "[sup] dog real-map recovery (CanFam, no PRDM9) $(date)"
cd=$CAMP/configs/real_dog
$FVPY scripts/bench.py gen --config "$cd" --demography realmap --species CanFam \
    --genetic-map Campbell2016_CanFam3_1 --Ne 13000 --mu 4e-9 --n-dip 10 \
    --seq-len 2000000 --n-regions 20 --seed 100 > "$CAMP/logs/gen_real_dog.log" 2>&1
$FVPY scripts/bench.py fastrho --config "$cd" --checkpoint "$CKPT" --stats "$STATS" --device "$DEV" >> "$CAMP/logs/fastrho_real_dog.log" 2>&1
$PVPY scripts/run_pyrho_config.py "$cd" > "$CAMP/logs/pyrho_real_dog.log" 2>&1 && \
  $FVPY scripts/bench.py ingest --config "$cd" --kind pyrho >> "$CAMP/logs/pyrho_real_dog.log" 2>&1
$FVPY scripts/bench.py score --config "$cd" --methods fastrho pyrho --results "$R" > "$CAMP/logs/score_real_dog.log" 2>&1

echo "[sup] pyrho wrong-demography panel (bottleneck data, constant-Ne table) $(date)"
wd=$CAMP/configs/bottleneck_n20_wd
if [ -d "$CAMP/configs/bottleneck_n20" ]; then
  rm -rf "$wd"; cp -r "$CAMP/configs/bottleneck_n20" "$wd"
  $FVPY -c "import json;p='$wd/config.json';c=json.load(open(p));c['name']='bottleneck_n20_wd';c['popsizes']=[c['Ne']];c['epochtimes']=[];json.dump(c,open(p,'w'))"
  rm -f "$wd"/region_*.rmap "$wd"/pred_*.npz
  $PVPY scripts/run_pyrho_config.py "$wd" > "$CAMP/logs/pyrho_wd.log" 2>&1 && \
    $FVPY scripts/bench.py ingest --config "$wd" --kind pyrho >> "$CAMP/logs/pyrho_wd.log" 2>&1
  $FVPY scripts/bench.py fastrho --config "$wd" --checkpoint "$CKPT" --stats "$STATS" --device "$DEV" >> "$CAMP/logs/fastrho_wd.log" 2>&1
  $FVPY scripts/bench.py score --config "$wd" --methods fastrho pyrho --results "$R" > "$CAMP/logs/score_wd.log" 2>&1
fi

echo "[sup] between-population difference + noise floor $(date)"
$FVPY scripts/between_pop.py --checkpoint "$CKPT" --stats "$STATS" --device "$DEV" \
    --diff-frac 0.5 --out "$R/between_pop_d50.json" > "$CAMP/logs/between_d50.log" 2>&1
$FVPY scripts/between_pop.py --checkpoint "$CKPT" --stats "$STATS" --device "$DEV" \
    --diff-frac 0.0 --out "$R/between_pop_d00.json" > "$CAMP/logs/between_d00.log" 2>&1

echo "[sup] held-out evaluation (coverage / calibration) $(date)"
$FVPY -m fastrho.evaluate --checkpoint "$CKPT" --stats "$STATS" \
    --shards "$CAMP/shards15k/test" --device "$DEV" > "$CAMP/logs/heldout.log" 2>&1 || true

$FVPY scripts/collate.py "$R" > "$R/summary.json" 2>> "$CAMP/logs/collate.log"
echo "[sup] DONE $(date)"
