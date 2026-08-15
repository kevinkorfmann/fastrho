#!/usr/bin/env bash
set -euo pipefail

analysis_parent=/home/kkor/fastrho_iain_20260814
analysis_root=$analysis_parent/matched_complete
analysis_code=/home/kkor/fastrho
analysis_python=/home/kkor/venvs/fastrho/bin/python
analysis_checkpoint=$(/usr/bin/sed -n '1p' /home/kkor/fastrho_data/campaign_dog_bottleneck/ckpt.txt)
analysis_stats=/home/kkor/fastrho_data/campaign_dog_bottleneck/shards15k/feat_stats.npz
export FASTRHO_CODE=$analysis_code
export PYTHONNOUSERSITE=1
mkdir -p "$analysis_root"/{hap,maps,logs,results}

if ! test -f "$analysis_root/matching_manifest.json"; then
  "$analysis_python" "$analysis_parent/code/canid_marker_match.py" prepare \
    --wolf "$analysis_parent/complete_inputs/wolf33_chr1_complete.npz" \
    --dog "$analysis_parent/complete_inputs/dog42_chr1_complete.npz" \
    --output "$analysis_root" --replicates 20 --seed 20260814 \
    >"$analysis_root/logs/prepare.log" 2>&1
fi

while ! test -f "$analysis_parent/MULTICHROM_INFER_DONE"; do
  sleep 30
done

even=(0 2 4 6 8 10 12 14 16 18)
odd=(1 3 5 7 9 11 13 15 17 19)
CUDA_VISIBLE_DEVICES=0 "$analysis_python" "$analysis_parent/code/canid_batch_infer.py" \
  --root "$analysis_root" --checkpoint "$analysis_checkpoint" --stats "$analysis_stats" \
  --device cuda:0 --indices "${even[@]}" >"$analysis_root/logs/matched_gpu0.log" 2>&1 &
pid_zero=$!
CUDA_VISIBLE_DEVICES=1 "$analysis_python" "$analysis_parent/code/canid_batch_infer.py" \
  --root "$analysis_root" --checkpoint "$analysis_checkpoint" --stats "$analysis_stats" \
  --device cuda:0 --indices "${odd[@]}" >"$analysis_root/logs/matched_gpu1.log" 2>&1 &
pid_one=$!
wait "$pid_zero"
wait "$pid_one"

"$analysis_python" "$analysis_parent/code/canid_marker_match.py" summarize \
  --root "$analysis_root" --replicates 20 --bootstrap 10000 --seed 20260814 \
  --output "$analysis_root/results/canid_marker_matched_complete.json" \
  --table "$analysis_root/results/canid_marker_matched_complete_long.csv" \
  >"$analysis_root/logs/matched_summary.log" 2>&1
touch "$analysis_root/COMPLETE_MATCH_DONE" "$analysis_parent/COMPLETE_MATCH_DONE"
