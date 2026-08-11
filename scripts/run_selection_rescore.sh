#!/usr/bin/env bash
# Repair config.json for every slim_dr condition, then (re-)run fastrho inference + scoring on the
# already-generated trees. No regeneration. Runs on sesame.
set -uo pipefail
export PYTHONNOUSERSITE=1
ROOT=/home/kkor/fastrho_data/slim_dr
CKPT=/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt
STATS=/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz
PY=/home/kkor/venvs/fastrho/bin/python
RESULTS=$ROOT/results
cd /home/kkor/fastrho
mkdir -p "$RESULTS"

for d in "$ROOT"/*/; do
  name=$(basename "$d")
  [ -f "$d/config.json" ] || continue
  [ "$name" = "results" ] && continue
  echo "=== $name ==="
  $PY scripts/fix_selection_config.py "$d" 2>&1 | tail -1
  $PY scripts/bench.py fastrho --config "$d" --checkpoint "$CKPT" --stats "$STATS" --device cuda:0 2>&1 | tail -1
  $PY scripts/bench.py score --config "$d" --methods fastrho --grids 25000 100000 500000 --results "$RESULTS" 2>&1 | tail -1
done
echo "ALL_DONE_RESCORE"
