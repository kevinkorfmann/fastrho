#!/usr/bin/env bash
# Dose-response + sweep-modes expansion of the linked-selection stress test (runs on sesame).
# Generates SLiM regions across a sweep-strength series, a BGS-intensity series, and extra sweep
# modes; infers each with the FROZEN base model (no retraining); scores at 25/100/500 kb.
set -uo pipefail
export PYTHONNOUSERSITE=1

ROOT=/home/kkor/fastrho_data/slim_dr
CKPT=/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt
STATS=/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz
PY=/home/kkor/venvs/fastrho/bin/python
NREG=${1:-30}
NPROC=${2:-30}
RESULTS=$ROOT/results
cd /home/kkor/fastrho
mkdir -p "$RESULTS"

run() {  # $1=name $2=mode ; selection env vars set by caller
  local name=$1 mode=$2 dir=$ROOT/$1
  echo "=== GEN $name ($mode) ==="
  rm -rf "$dir"
  SLIM_NAME=$name $PY scripts/slim_gen.py "$dir" "$mode" "$NREG" 0 "$NPROC" 2>&1 | tail -1
  $PY scripts/bench.py fastrho --config "$dir" --checkpoint "$CKPT" --stats "$STATS" --device cuda:0 2>&1 | tail -1
  $PY scripts/bench.py score --config "$dir" --methods fastrho --grids 25000 100000 500000 --results "$RESULTS" 2>&1 | tail -1
}

# neutral baseline
run neutral neutral
# sweep-strength series (2Ns = 100,200,500,1000,2000 at Ne=1e4): pure sweep on a neutral background
for s in 0.005 0.01 0.025 0.05 0.1; do
  tag=${s#0.}
  SLIM_SWEEP_S=$s run sw_s$tag sweep
done
# BGS-intensity series: vary exonic fraction under purifying selection
for ef in 0.10 0.25 0.50; do
  tag=${ef#0.}
  SLIM_EXON_FRAC=$ef run bgs_ef$tag bgs
done
# extra sweep modes (s=0.05): soft (5 origins) and partial (stops at freq 0.7)
SLIM_SWEEP_S=0.05 SLIM_SOFT_K=5 run sw_soft sweep
SLIM_SWEEP_S=0.05 SLIM_SWEEP_TARGET=0.7 run sw_partial sweep

echo "ALL_DONE_SELECTION_DR"
ls "$RESULTS"
