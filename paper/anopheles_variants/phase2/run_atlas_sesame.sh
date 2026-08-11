#!/usr/bin/env bash
set -euo pipefail

phase2_root="${1:-data/ag1000g-phase2-ar1}"
repo="${2:-$(git rev-parse --show-toplevel)}"
extract_python="${FASTRHO_PHASE2_EXTRACT_PYTHON:-python3}"
inference_python="${FASTRHO_PHASE2_INFERENCE_PYTHON:-python3}"
source_dir="${FASTRHO_PHASE2_SOURCE_DIR:-$repo/paper/anopheles_variants/common}"
raw="$phase2_root/raw/haplotypes_main_hdf5"
selected="${FASTRHO_PHASE2_SELECTED_SAMPLES:-$repo/paper/anopheles_variants/phase2/cohorts/selected_samples.tsv}"
normalized="$phase2_root/normalized"
maps="$phase2_root/maps"
logs="$phase2_root/logs"
checkpoint="${FASTRHO_PHASE2_CHECKPOINT:-$repo/downloaded-models/high-ne-v1/model.ckpt}"
stats="${FASTRHO_PHASE2_STATS:-$repo/downloaded-models/high-ne-v1/feat_stats.npz}"

test -f "$phase2_root/provenance/raw_files.sha256"
test -f "$selected"
test -f "$checkpoint"
test -f "$stats"
mkdir -p "$normalized" "$maps" "$logs"

extract_pids=()
for arm in 2R 2L 3R 3L X; do
  "$extract_python" "$source_dir/extract_phase2_hdf5.py" \
    --source "$raw/ag1000g.phase2.ar1.haplotypes.${arm}.h5" \
    --arm "$arm" \
    --selected-samples "$selected" \
    --out "$normalized" \
    > "$logs/extract_${arm}.log" 2>&1 &
  extract_pids+=("$!")
done
for pid in "${extract_pids[@]}"; do
  wait "$pid"
done

normalized_count=$(find "$normalized" -maxdepth 1 -type f -name '*.h5' | wc -l | tr -d ' ')
test "$normalized_count" = 45
find "$normalized" -maxdepth 1 -type f \( -name '*.h5' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 shasum -a 256 > "$phase2_root/provenance/normalized_files.sha256"

cd "$repo"
CUDA_VISIBLE_DEVICES=0 "$inference_python" "$source_dir/infer_phase2.py" \
  --input "$normalized" --checkpoint "$checkpoint" --stats "$stats" \
  --out "$maps" --device cuda:0 --shard 0 --nshards 2 \
  > "$logs/infer_0.log" 2>&1 &
infer0=$!
CUDA_VISIBLE_DEVICES=1 "$inference_python" "$source_dir/infer_phase2.py" \
  --input "$normalized" --checkpoint "$checkpoint" --stats "$stats" \
  --out "$maps" --device cuda:0 --shard 1 --nshards 2 \
  > "$logs/infer_1.log" 2>&1 &
infer1=$!
wait "$infer0"
wait "$infer1"

map_count=$(find "$maps" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')
test "$map_count" = 45
find "$maps" -maxdepth 1 -type f \( -name '*.npz' -o -name '*.json' \) -print0 \
  | sort -z | xargs -0 shasum -a 256 > "$phase2_root/provenance/map_files.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$phase2_root/provenance/atlas_completed_utc.txt"
