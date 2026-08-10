#!/usr/bin/env bash
# Phase 1 driver (runs on sesame): generate SLiM linked-selection regions for three regimes,
# infer with the FROZEN base model (no retraining), score, and collate selection.json.
set -euo pipefail
export PYTHONNOUSERSITE=1

ROOT=/home/kkor/fastrho_data/slim
CKPT=/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt
STATS=/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz
PY=/home/kkor/venvs/fastrho/bin/python
NREG=${1:-24}
NPROC=${2:-24}
RESULTS=$ROOT/results
cd /home/kkor/fastrho
mkdir -p "$RESULTS"

for MODE in neutral bgs sweep; do
  echo "=== GENERATE $MODE ($NREG regions) ==="
  rm -rf "$ROOT/$MODE"
  $PY scripts/slim_gen.py "$ROOT/$MODE" "$MODE" "$NREG" 0 "$NPROC"
  echo "=== INFER $MODE (frozen base model) ==="
  $PY scripts/bench.py fastrho --config "$ROOT/$MODE" --checkpoint "$CKPT" --stats "$STATS" --device cuda:0
  echo "=== SCORE $MODE ==="
  $PY scripts/bench.py score --config "$ROOT/$MODE" --methods fastrho \
      --grids 25000 100000 500000 --results "$RESULTS"
done

echo "=== COLLATE selection.json ==="
$PY - "$RESULTS" <<'PYEOF'
import json, os, sys
rd = sys.argv[1]
out = {}
for mode in ("neutral", "bgs", "sweep"):
    with open(os.path.join(rd, "slim_%s.json" % mode)) as fh:
        out["slim_" + mode] = json.load(fh)
with open(os.path.join(rd, "selection.json"), "w") as fh:
    json.dump(out, fh, indent=2)
# console summary
for mode in ("neutral", "bgs", "sweep"):
    r = out["slim_" + mode]
    cells = " ".join("%s=%.3f" % (s, r["scales"][s]["fastrho"]["pearson"])
                     for s in ("25kb", "100kb", "500kb"))
    print("slim_%-8s n_regions=%d  %s" % (mode, r["n_regions_scored"], cells))
print("wrote", os.path.join(rd, "selection.json"))
PYEOF
echo "ALL_DONE_SELECTION"
