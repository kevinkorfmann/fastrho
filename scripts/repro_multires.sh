#!/usr/bin/env bash
# Render the 6-panel low-res/high-res "multires" figure (each tool at 100 kb then 25 kb)
# from an EXISTING repro run -- no GPU / no pyrho recompute. It only re-runs the cheap
# `score` step (which now also writes the 25-kb high-res arrays) and then `multires`.
#
# Prereq: scripts/repro_relernn.sh has been run once, so $REPRO holds the per-method
# outputs (fastrho_pred.npz, region_000.rmap, relernn_proj/*PREDICT*txt). If they are
# gone, run scripts/repro_relernn.sh first (it now emits the multires figure too).
#
#   bash scripts/repro_multires.sh
set -uo pipefail

REPRO=/home/kkor/fastrho/repro_relernn
FVPY=/home/kkor/venvs/fastrho/bin/python
export PYTHONNOUSERSITE=1
cd /home/kkor/fastrho

fail() { echo "FAILED: $1"; exit 1; }

# locate the existing per-method outputs (all produced by repro_relernn.sh)
[ -s "$REPRO/region_000.npz" ]     || fail "no $REPRO/region_000.npz -- run scripts/repro_relernn.sh first"
[ -s "$REPRO/fastrho_pred.npz" ]   || fail "no fastrho_pred.npz -- run scripts/repro_relernn.sh first"
[ -s "$REPRO/region_000.rmap" ]    || echo "note: no pyrho region_000.rmap -- pyrho panels will be skipped"
PRED=$(ls "$REPRO"/relernn_proj/*PREDICT*txt 2>/dev/null | head -1)
[ -s "$PRED" ] || echo "note: no ReLERNN PREDICT.txt -- ReLERNN panels will be skipped"

# re-score: rebins the existing outputs to BOTH 100 kb and 25 kb and rewrites the bundle
echo "=== re-score (adds 25-kb high-res arrays to repro_showdown.npz) ==="
PYTHONNOUSERSITE=1 "$FVPY" scripts/repro_relernn_fig2.py score --data "$REPRO" \
  ${PRED:+--predict "$PRED"} --metrics "$REPRO/repro_metrics.json" \
  --showdown "$REPRO/repro_showdown.npz" 2>&1 | tee "$REPRO/05_score.log" || fail "score"

# render the 6-panel multires figure
echo "=== multires plot ==="
PYTHONNOUSERSITE=1 "$FVPY" scripts/repro_relernn_fig2.py multires \
  --showdown "$REPRO/repro_showdown.npz" \
  --png "$REPRO/repro_relernn_multires.png" 2>&1 | tee "$REPRO/07_multires.log" || fail "multires"

echo "=== DONE ==="
echo "copy back to the repo with:"
echo "  scp sesame:$REPRO/repro_showdown.npz                paper/figdata/"
echo "  scp sesame:$REPRO/repro_relernn_multires.{pdf,png}  paper/figures/"
