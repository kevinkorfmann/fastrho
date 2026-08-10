#!/bin/bash
set -euo pipefail

ROOT=/home/kkor/realdata
CODE=/home/kkor/fastrho
NUMPY_PY=/home/kkor/miniconda3/envs/gpu_tsinfer/bin/python
MODEL_PY=/home/kkor/venvs/fastrho/bin/python
CHECKPOINT=$(cat /home/kkor/fastrho_data/campaign_dog_bottleneck/ckpt.txt)
STATS=/home/kkor/fastrho_data/campaign_dog_bottleneck/shards15k/feat_stats.npz
STRUCTURE_CHECKPOINT=/home/kkor/fastrho_data/campaign_wolf_structure/train/fastrho/version_0/checkpoints/epoch=11-val_pearson=0.772.ckpt
STRUCTURE_STATS=/home/kkor/fastrho_data/campaign_wolf_structure/shards/feat_stats.npz

while pgrep -f 'canid_multi_panel_from_vcf.py.*_ref_chunk' >/dev/null; do
  sleep 30
done

dog_chunks=()
wolf_chunks=()
for index in 01 02 03 04 05 06 07 08; do
  dog="$ROOT/hap/dog67_ref_chunk${index}.npz"
  wolf="$ROOT/hap/wolf18_ref_chunk${index}.npz"
  test -s "$dog"
  test -s "$wolf"
  dog_chunks+=("$dog")
  wolf_chunks+=("$wolf")
done

"$NUMPY_PY" "$CODE/scripts/canid_merge_npz.py" \
  "$ROOT/hap/dog67_full_reference.npz" "${dog_chunks[@]}" \
  --key dog67_full_reference
"$NUMPY_PY" "$CODE/scripts/canid_merge_npz.py" \
  "$ROOT/hap/wolf_china18_full_reference.npz" "${wolf_chunks[@]}" \
  --key wolf_china18_full_reference
"$NUMPY_PY" "$CODE/scripts/canid_npz_subset.py" \
  "$ROOT/hap/dog67_full_reference.npz" "$ROOT/dog_china_ids.txt" \
  "$ROOT/dog_village42_ids.txt" "$ROOT/hap/dog_village42_full_reference.npz" \
  --key dog_village42_full_reference

"$MODEL_PY" "$CODE/scripts/wolf_subset_infer.py" \
  dog67_full_reference "$CHECKPOINT" "$STATS" \
  "$ROOT/maps/dog67_full_reference_dogbn.npz" cuda:0
"$MODEL_PY" "$CODE/scripts/wolf_subset_infer.py" \
  wolf_china18_full_reference "$CHECKPOINT" "$STATS" \
  "$ROOT/maps/wolf_china18_full_reference_dogbn.npz" cuda:0
"$MODEL_PY" "$CODE/scripts/wolf_subset_infer.py" \
  dog_village42_full_reference "$CHECKPOINT" "$STATS" \
  "$ROOT/maps/dog_village42_full_reference_dogbn.npz" cuda:0
"$MODEL_PY" "$CODE/scripts/wolf_subset_infer.py" \
  wolf_china18_full_reference "$STRUCTURE_CHECKPOINT" "$STRUCTURE_STATS" \
  "$ROOT/maps/wolf_china18_full_reference_specialist.npz" cuda:0
