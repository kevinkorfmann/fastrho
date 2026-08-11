#!/usr/bin/env bash
# Densely-sampled dose-response for the linked-selection figure (runs on sesame).
# Many more x-points than the first pass: ~9 sweep strengths, 7 BGS intensities, and a
# continuous sweep-completion series. Frozen base model, no retraining; scored at 25/100/500 kb.
set -uo pipefail
export PYTHONNOUSERSITE=1

ROOT=/home/kkor/fastrho_data/slim_dr2
CKPT=/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt
STATS=/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz
PY=/home/kkor/venvs/fastrho/bin/python
NREG=${1:-40}
NPROC=${2:-40}
RESULTS=$ROOT/results
cd /home/kkor/fastrho
mkdir -p "$RESULTS"

run() {  # $1=name $2=mode ; selection env set by caller
  local name=$1 mode=$2 dir=$ROOT/$1
  echo "=== GEN $name ($mode) ==="
  rm -rf "$dir"
  SLIM_NAME=$name $PY scripts/slim_gen.py "$dir" "$mode" "$NREG" 0 "$NPROC" 2>&1 | tail -1
  $PY scripts/bench.py fastrho --config "$dir" --checkpoint "$CKPT" --stats "$STATS" --device cuda:0 2>&1 | tail -1
  $PY scripts/bench.py score --config "$dir" --methods fastrho --grids 25000 100000 500000 --results "$RESULTS" 2>&1 | tail -1
}

run neutral neutral

# sweep-strength series (hard sweeps to fixation): 2Ne*s = 50 .. 4000
i=0
for s in 0.0025 0.005 0.0075 0.0125 0.02 0.035 0.06 0.1 0.2; do
  i=$((i+1)); SLIM_SWEEP_S=$s run swstr_$i sweep
done

# background-selection intensity series: exonic fraction 5% .. 60%
i=0
for ef in 0.05 0.10 0.15 0.22 0.32 0.45 0.60; do
  i=$((i+1)); SLIM_EXON_FRAC=$ef run bgsint_$i bgs
done

# sweep-completion series (fixed s=0.05): final beneficial-allele frequency 0.2 .. 1.0
i=0
for t in 0.2 0.35 0.5 0.65 0.8 0.9 1.0; do
  i=$((i+1)); SLIM_SWEEP_S=0.05 SLIM_SWEEP_TARGET=$t run compl_$i sweep
done

echo "ALL_DONE_DR2"
