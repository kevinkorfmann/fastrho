#!/usr/bin/env bash
# Reproduce ReLERNN Fig 2A on sesame AND run the three-method showdown on the identical data:
#   data (fastrho venv) -> fastrho forward pass (fastrho venv, GPU)
#   -> pyrho table+optimize (pyrho venv, CPU) -> ReLERNN full pipeline (relernn2 venv, GPU)
#   -> score + figdata (fastrho venv) -> plot (fastrho venv). Writes REPRO_DONE / REPRO_FAIL.
#
#   CUDA=0 bash scripts/repro_relernn.sh
set -uo pipefail

REPRO=/home/kkor/fastrho/repro_relernn
FVPY=/home/kkor/venvs/fastrho/bin/python
RVPY=/home/kkor/venvs/relernn2/bin/python
PVPY=/home/kkor/venvs/pyrho/bin/python
UV=/home/kkor/.local/bin/uv
export PYTHONNOUSERSITE=1
cd /home/kkor/fastrho

mkdir -p "$REPRO"
rm -f "$REPRO/REPRO_DONE" "$REPRO/REPRO_FAIL"
fail() { echo "STAGE FAILED: $1"; touch "$REPRO/REPRO_FAIL"; exit 1; }

# stage 1: build Comeron 2L dataset (skip if already present). config.json carries the
# pyrho size-history fields (n_dip/popsizes) so run_pyrho_config can build a matched table.
if [ ! -s "$REPRO/region_000.vcf" ]; then
  echo "=== stage 1: data ==="
  PYTHONNOUSERSITE=1 "$FVPY" scripts/repro_relernn_fig2.py data --out "$REPRO" \
    2>&1 | tee "$REPRO/01_data.log" || fail "data"
else
  echo "=== stage 1: data (cached) ==="
fi

# stage 2: fastrho -- single forward pass of the frozen base model on the identical VCF
echo "=== stage 2: fastrho (frozen model, 1 forward pass) ==="
CUDA_VISIBLE_DEVICES="${CUDA:-0}" PYTHONNOUSERSITE=1 "$FVPY" \
  scripts/repro_relernn_fig2.py fastrho --data "$REPRO" --device cuda:0 \
  2>&1 | tee "$REPRO/02_fastrho.log" || fail "fastrho"

# stage 3: pyrho -- Ne=2.5e5 ldpop table + optimize on the identical VCF -> region_000.rmap
echo "=== stage 3: pyrho (matched ldpop table) ==="
PYTHONNOUSERSITE=1 "$PVPY" scripts/run_pyrho_config.py "$REPRO" \
  2>&1 | tee "$REPRO/03_pyrho.log" || fail "pyrho"

# stage 4: ReLERNN SIMULATE->TRAIN->PREDICT, upRTR auto-tuned from the true map peak
echo "=== stage 4: ReLERNN (auto upRTR) ==="
CUDA_VISIBLE_DEVICES="${CUDA:-0}" "$RVPY" scripts/run_relernn_config.py "$REPRO" auto \
  2>&1 | tee "$REPRO/04_relernn.log" || fail "relernn"

PRED=$(ls "$REPRO"/relernn_proj/*PREDICT*txt 2>/dev/null | head -1)
[ -s "$PRED" ] || fail "no PREDICT.txt produced"
echo "PREDICT: $PRED"

# stage 5: score all available methods -> per-method 100kb metrics + committed figdata bundle
echo "=== stage 5: score (3-method, 100kb) ==="
"$UV" pip install --python "$FVPY" -q matplotlib 2>/dev/null || true
PYTHONNOUSERSITE=1 "$FVPY" scripts/repro_relernn_fig2.py score --data "$REPRO" \
  --predict "$PRED" --metrics "$REPRO/repro_metrics.json" \
  --showdown "$REPRO/repro_showdown.npz" 2>&1 | tee "$REPRO/05_score.log" || fail "score"

# stage 6: render the 3-panel 100-kb figure from the figdata bundle (paper_style)
echo "=== stage 6: plot (3-panel, 100 kb) ==="
PYTHONNOUSERSITE=1 "$FVPY" scripts/repro_relernn_fig2.py plot \
  --showdown "$REPRO/repro_showdown.npz" \
  --png "$REPRO/repro_relernn_fig2a.png" 2>&1 | tee "$REPRO/06_plot.log" || fail "plot"

# stage 7: render the 6-panel low-res/high-res multires figure from the SAME bundle
# (score now also stores the 25-kb high-res arrays; each tool gets a 100-kb then a 25-kb panel)
echo "=== stage 7: multires plot (6 panels: each tool at 100 kb then 25 kb) ==="
PYTHONNOUSERSITE=1 "$FVPY" scripts/repro_relernn_fig2.py multires \
  --showdown "$REPRO/repro_showdown.npz" \
  --png "$REPRO/repro_relernn_multires.png" 2>&1 | tee "$REPRO/07_multires.log" || fail "multires"

echo "=== DONE ==="
echo "copy figdata + figures to the repo with:"
echo "  scp sesame:$REPRO/repro_showdown.npz  paper/figdata/"
echo "  scp sesame:$REPRO/repro_metrics.json  paper/figdata/"
echo "  scp sesame:$REPRO/repro_relernn_fig2a.{pdf,png}   paper/figures/"
echo "  scp sesame:$REPRO/repro_relernn_multires.{pdf,png} paper/figures/"
touch "$REPRO/REPRO_DONE"
