#!/usr/bin/env bash
set -euo pipefail

phase2_root="${1:-/home/kkor/agam_phase2}"
repo="${2:-/home/kkor/fastrho}"
nworkers="${FASTRHO_PHASE2_WORKERS:-8}"
python="${FASTRHO_PHASE2_INFERENCE_PYTHON:-/home/kkor/venvs/fastrho/bin/python}"
checkpoint="/home/kkor/fastrho_data/campaign_hidip/train15k/fastrho/version_0/checkpoints/epoch=37-val_loss=-0.178.ckpt"
stats="/home/kkor/fastrho_data/campaign_hidip/shards15k/feat_stats.npz"
input="$phase2_root/normalized"
maps="$phase2_root/maps"
logs="$phase2_root/logs"

test "$(find "$input" -maxdepth 1 -type f -name '*.h5' | wc -l | tr -d ' ')" = 45
mkdir -p "$maps" "$logs"
cd "$repo"
pids=()
for ((shard=0; shard<nworkers; shard++)); do
  gpu=$((shard % 2))
  CUDA_VISIBLE_DEVICES="$gpu" "$python" "$phase2_root/source/infer_phase2.py" \
    --input "$input" --checkpoint "$checkpoint" --stats "$stats" \
    --out "$maps" --device cuda:0 --shard "$shard" --nshards "$nworkers" \
    > "$logs/infer_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

test "$(find "$maps" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')" = 45
find "$maps" -maxdepth 1 -type f \( -name '*.npz' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 shasum -a 256 > "$phase2_root/provenance/map_files.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$phase2_root/provenance/atlas_completed_utc.txt"
