#!/usr/bin/env bash
# pyrho-under-selection baseline (runs on sesame, after run_selection_dr.sh).
# Runs pyrho with a NEUTRAL constant-Ne table -- unaware of selection, the realistic baseline -- on
# the same SLiM regions, then re-scores fastrho AND pyrho against the truth. Shows fastrho degrades
# less than the demography/selection-agnostic composite-likelihood under linked selection.
set -uo pipefail
export PYTHONNOUSERSITE=1

ROOT=${SLIM_DR_ROOT:-/home/kkor/fastrho_data/slim_dr}
RESULTS=$ROOT/results
PYRHO_PY=/home/kkor/venvs/pyrho/bin/python
PY=/home/kkor/venvs/fastrho/bin/python
cd /home/kkor/fastrho

# neutral + the sweep-strength series + the BGS-intensity series (matched fastrho-vs-pyrho curves).
# Pass a space-separated condition list as $1 to override (e.g. to run only the missing ones).
CONDS="${1:-neutral sw_s005 sw_s01 sw_s025 sw_s05 sw_s1 bgs_ef10 bgs_ef25 bgs_ef50}"

for name in $CONDS; do
  d=$ROOT/$name
  [ -d "$d" ] || { echo "skip $name (no dir)"; continue; }
  echo "=== PYRHO $name ==="
  $PYRHO_PY scripts/run_pyrho_config.py "$d" 2>&1 | tail -1
  $PY scripts/bench.py ingest --config "$d" --kind pyrho 2>&1 | tail -1
  $PY scripts/bench.py score --config "$d" --methods fastrho pyrho \
      --grids 25000 100000 500000 --results "$RESULTS" 2>&1 | tail -2
done
echo "ALL_DONE_SELECTION_PYRHO"
