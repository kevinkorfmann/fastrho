#!/bin/bash
set -euo pipefail

ROOT=/home/kkor/realdata
CODE=/home/kkor/fastrho
VCF=https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz
NUMPY_PY=/home/kkor/miniconda3/envs/gpu_tsinfer/bin/python
MODEL_PY=/home/kkor/venvs/fastrho/bin/python
CHECKPOINT=$(cat /home/kkor/fastrho_data/campaign_dog_bottleneck/ckpt.txt)
STATS=/home/kkor/fastrho_data/campaign_dog_bottleneck/shards15k/feat_stats.npz
IDS=$ROOT/wolf33_ids.txt

regions=(
  chr1:1-15000000
  chr1:15000001-30000000
  chr1:30000001-45000000
  chr1:45000001-60000000
  chr1:60000001-75000000
  chr1:75000001-90000000
  chr1:90000001-105000000
  chr1:105000001-123000000
)

pids=()
chunks=()
for offset in 0 1 2 3 4 5 6 7; do
  index=$(printf '%02d' "$((offset + 1))")
  output=$ROOT/hap/wolf33_ref_chunk${index}.npz
  log=$ROOT/logs/wolf33_ref_chunk${index}.log
  chunks+=("$output")
  if test -s "$output"; then
    continue
  fi
  mkdir -p "$ROOT/logs"
  "$NUMPY_PY" "$CODE/scripts/canid_multi_panel_from_vcf.py" "$VCF" \
    --panel wolf33 "$IDS" "$output" \
    --region "${regions[$offset]}" --missing-policy reference >"$log" 2>&1 &
  pids+=("$!")
  if test "${#pids[@]}" -eq 4; then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$NUMPY_PY" "$CODE/scripts/canid_merge_npz.py" \
  "$ROOT/hap/wolf33_full_reference.npz" "${chunks[@]}" \
  --key wolf33_full_reference

CUDA_VISIBLE_DEVICES=0 "$MODEL_PY" "$CODE/scripts/wolf_subset_infer.py" \
  wolf33_full_reference "$CHECKPOINT" "$STATS" \
  "$ROOT/maps/wolf33_full_reference_dogbn.npz" cuda:0

"$NUMPY_PY" "$CODE/scripts/canid_empirical_bootstrap.py" \
  --wolf "$ROOT/maps/wolf33_full_reference_dogbn.npz" \
  --dog all_dogs "$ROOT/maps/dog67_full_reference_dogbn.npz" \
  --dog village_dogs "$ROOT/maps/dog_village42_full_reference_dogbn.npz" \
  --output "$ROOT/maps/wolf33_full_reference_bootstrap.json"
