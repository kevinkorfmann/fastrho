#!/usr/bin/env bash
# Overnight fastrho vs pyrho vs ReLERNN campaign on sesame (GPU 0; avoid GPU 1 = user job).
# Trains the paper model, runs the comprehensive benchmark + real-map recovery, collates
# results, and writes ALL_DONE. Non-critical per-config failures are logged, not fatal.
export PYTHONNOUSERSITE=1 PYTHONWARNINGS=ignore
FVPY=/home/kkor/venvs/fastrho/bin/python
PVPY=/home/kkor/venvs/pyrho/bin/python
RVPY=/home/kkor/venvs/relernn2/bin/python
cd /home/kkor/fastrho

CAMP=/home/kkor/fastrho_data/campaign
M=$CAMP/markers; R=$CAMP/results; LOG=$CAMP/logs
mkdir -p "$M" "$R" "$LOG" "$CAMP/configs"
mark(){ touch "$M/$1"; echo "[$(date)] $1"; }

# name|demography|n_dip|genetic_map|relernn_full|seq_len|n_regions
CONFIGS=(
 "const_n20|constant|10|-|1|2000000|24"
 "const_n40|constant|20|-|1|2000000|24"
 "real_hapmap|realmap|10|HapMapII_GRCh38|1|2000000|24"
 "real_decode|realmap|10|DeCodeSexAveraged_GRCh38|1|2000000|24"
 "bottleneck_n20|bottleneck|10|-|0|2000000|24"
 "expansion_n20|expansion|10|-|0|2000000|24"
 "const_n100|constant|50|-|0|2000000|24"
)

# ---------- Stage A: big training set ----------
mark STAGE_A_START
$FVPY -m fastrho.simulate --data-dir "$CAMP/train_sims" --num-ts 6000 --num-processes 40 --sequence-length 1000000 > "$LOG/sim_train.log" 2>&1
$FVPY -m fastrho.simulate --data-dir "$CAMP/test_sims"  --num-ts 300  --num-processes 40 --sequence-length 1000000 > "$LOG/sim_test.log" 2>&1
$FVPY -m fastrho.preprocess --sim-dir "$CAMP/train_sims" --out-dir "$CAMP/shards/train" --with-features --num-processes 40 > "$LOG/prep_train.log" 2>&1
$FVPY -m fastrho.preprocess --sim-dir "$CAMP/test_sims"  --out-dir "$CAMP/shards/test"  --with-features --num-processes 40 > "$LOG/prep_test.log" 2>&1
mark STAGE_A_DONE

# ---------- Stage C (background, CPU): gen + pyrho for every config ----------
(
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r name demo ndip gmap rfull slen nreg <<< "$c"
    cd="$CAMP/configs/$name"
    gmarg=""; [ "$gmap" != "-" ] && gmarg="--genetic-map $gmap"
    rfarg=""; [ "$rfull" = "1" ] && rfarg="--relernn-full"
    $FVPY scripts/bench.py gen --config "$cd" --demography "$demo" --n-dip "$ndip" \
        --seq-len "$slen" --n-regions "$nreg" --seed 100 $gmarg $rfarg > "$LOG/gen_$name.log" 2>&1
    $PVPY scripts/run_pyrho_config.py "$cd" > "$LOG/pyrho_$name.log" 2>&1
    $FVPY scripts/bench.py ingest --config "$cd" --kind pyrho >> "$LOG/pyrho_$name.log" 2>&1
    touch "$M/CFG_PREP_$name"
  done
  mark STAGE_C_DONE
) &
CPID=$!

# ---------- Stage B: train the paper model (GPU 0) ----------
mark STAGE_B_START
$FVPY -m fastrho.train --model base --dataset-path "$CAMP/shards" --epochs 50 --gpus 0 \
    --batch-size 48 --lr 4e-4 --workers 10 --log-dir "$CAMP/train" > "$LOG/train.log" 2>&1
mark STAGE_B_DONE
CKPT=$(ls -t "$CAMP"/train/fastrho/version_*/checkpoints/*.ckpt | head -1)
STATS="$CAMP/shards/feat_stats.npz"
echo "$CKPT" > "$CAMP/ckpt.txt"

wait $CPID    # ensure gen+pyrho prep finished

# ---------- fastrho predict on every config (the one amortized model) ----------
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name demo ndip gmap rfull slen nreg <<< "$c"
  $FVPY scripts/bench.py fastrho --config "$CAMP/configs/$name" --checkpoint "$CKPT" \
      --stats "$STATS" --device cuda:0 > "$LOG/fastrho_$name.log" 2>&1
done
mark FASTRHO_DONE

# ---------- ReLERNN on headline configs (GPU 0, full budget) ----------
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name demo ndip gmap rfull slen nreg <<< "$c"
  [ "$rfull" = "1" ] || continue
  cd="$CAMP/configs/$name"
  if $RVPY scripts/run_relernn_config.py "$cd" > "$LOG/relernn_$name.log" 2>&1; then
    P=$(ls "$cd"/relernn_proj/*PREDICT*txt 2>/dev/null | head -1)
    [ -n "$P" ] && $FVPY scripts/bench.py ingest --config "$cd" --kind relernn --predict "$P" >> "$LOG/relernn_$name.log" 2>&1
  fi
  touch "$M/RELERNN_$name"
done
mark RELERNN_DONE

# ---------- score every config + collate ----------
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name demo ndip gmap rfull slen nreg <<< "$c"
  methods="fastrho pyrho"; [ "$rfull" = "1" ] && methods="fastrho pyrho relernn"
  $FVPY scripts/bench.py score --config "$CAMP/configs/$name" --methods $methods --results "$R" > "$LOG/score_$name.log" 2>&1
done
$FVPY scripts/collate.py "$R" > "$R/summary.json" 2> "$LOG/collate.log"
mark ALL_DONE
echo "[$(date)] CAMPAIGN COMPLETE"
