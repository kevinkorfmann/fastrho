#!/usr/bin/env bash
set -euo pipefail

analysis_root=/home/kkor/fastrho_iain_20260814
analysis_python=/home/kkor/venvs/fastrho/bin/python
analysis_script=$analysis_root/code/multichrom_infer.py
export FASTRHO_CODE=/home/kkor/fastrho_dr
export PYTHONNOUSERSITE=1

while ! test -f "$analysis_root/MULTICHROM_EXTRACT_DONE" || \
  test "$(find "$analysis_root/matched_v2/maps" -type f -name '*.npz' 2>/dev/null | wc -l)" -lt 40; do
  sleep 30
done

run_dmel() {
  CUDA_VISIBLE_DEVICES=0 DEV=cuda:0 "$analysis_python" "$analysis_script" \
    --input "$analysis_root/multichrom_hap/dmel__*.npz" \
    --output "$analysis_root/results/dmel_multichrom.json" --species "Drosophila melanogaster" \
    >"$analysis_root/logs/infer_dmel_multichrom.log" 2>&1
}

run_jewelwasp() {
  CUDA_VISIBLE_DEVICES=0 DEV=cuda:0 "$analysis_python" "$analysis_script" \
    --input "$analysis_root/multichrom_hap/jewelwasp__*.npz" \
    --output "$analysis_root/results/jewelwasp_multichrom.json" --species "Nasonia vitripennis" \
    >"$analysis_root/logs/infer_jewelwasp_multichrom.log" 2>&1
}

run_aspen() {
  CUDA_VISIBLE_DEVICES=1 DEV=cuda:0 "$analysis_python" "$analysis_script" \
    --input "$analysis_root/multichrom_hap/aspen__*.npz" \
    --output "$analysis_root/results/aspen_multichrom.json" --species "Populus tremula" \
    >"$analysis_root/logs/infer_aspen_multichrom.log" 2>&1
}

run_dmel &
pid_dmel=$!
run_jewelwasp &
pid_jewelwasp=$!
run_aspen &
pid_aspen=$!
wait "$pid_dmel"
wait "$pid_jewelwasp"
wait "$pid_aspen"
touch "$analysis_root/MULTICHROM_INFER_DONE"
