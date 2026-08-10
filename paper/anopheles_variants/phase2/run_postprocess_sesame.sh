#!/usr/bin/env bash
set -euo pipefail

root="${1:-/home/kkor/agam_phase2}"
repo="${2:-/home/kkor/fastrho}"
system_python="${FASTRHO_PHASE2_SYSTEM_PYTHON:-python3}"
fastrho_python="${FASTRHO_PHASE2_INFERENCE_PYTHON:-/home/kkor/venvs/fastrho/bin/python}"
checkpoint="/home/kkor/fastrho_data/campaign_hidip/train15k/fastrho/version_0/checkpoints/epoch=37-val_loss=-0.178.ckpt"
stats="/home/kkor/fastrho_data/campaign_hidip/shards15k/feat_stats.npz"
pyrho="/home/kkor/venvs/pyrho/bin/pyrho"
metadata="$root/raw/metadata/samples.meta.txt"
selection="$root/cohorts/selection.tsv"
selected="$root/cohorts/selected_samples.tsv"
maps="$root/maps"
normalized="$root/normalized"
results="$root/results"
logs="$root/logs"
source="$root/source"

test "$(find "$maps" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')" = 45
mkdir -p "$results" "$logs"

"$system_python" "$source/phase2_map_qc.py" \
  --maps "$maps" --selection "$selection" \
  --out "$results/phase2_map_qc.json" \
  > "$logs/phase2_map_qc.log" 2>&1

"$system_python" "$source/export_phase2_atlas.py" \
  --maps "$maps" --selection "$selection" --selected-samples "$selected" \
  --metadata "$metadata" --out "$root/atlas" \
  > "$logs/export_atlas.log" 2>&1 &
pids=("$!")

"$system_python" "$source/phase2_2la.py" \
  --hdf5 "$root/raw/haplotypes_main_hdf5/ag1000g.phase2.ar1.haplotypes.2L.h5" \
  --metadata "$metadata" --selected-samples "$selected" --selection "$selection" \
  --tag-snps "$root/provenance/karyotype_tag_snps.csv" --maps "$maps" \
  --out "$results/phase2_2la.json" \
  > "$logs/phase2_2la.log" 2>&1 &
pids+=("$!")

"$system_python" "$source/phase2_resistance.py" \
  --normalized "$normalized" --maps "$maps" --selection "$selection" \
  --panel-spec "$root/provenance/anopheles_resistance_panels.tsv" \
  --out "$results/phase2_resistance.json" \
  > "$logs/phase2_resistance.log" 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=0 "$fastrho_python" "$source/phase2_pyrho.py" \
  --normalized "$normalized" --maps "$maps" --checkpoint "$checkpoint" \
  --stats "$stats" --pyrho "$pyrho" --device cuda:0 \
  --out "$results/phase2_pyrho.json" \
  > "$logs/phase2_pyrho.log" 2>&1 &
pids+=("$!")

pedigree_stage=()
if [[ -f "$results/pedigree/phase2_pedigree_call_manifest.json" ]]; then
  pedigree_stage=(--score-only)
fi
"$fastrho_python" "$source/phase2_pedigree.py" \
  --shapeit "$root/raw/haplotypes_crosses_shapeit" \
  --metadata "$root/raw/metadata/cross.samples.meta.txt" \
  --maps "$maps" --selection "$selection" \
  --core-dir "$source/pedigree_core" --out "$results/pedigree" \
  "${pedigree_stage[@]}" \
  > "$logs/phase2_pedigree.log" 2>&1 &
pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid"
done

"$system_python" "$source/phase2_manuscript_data.py" \
  --results "$results" --selection "$selection" \
  --out "$results/manuscript_generated" \
  > "$logs/phase2_manuscript_data.log" 2>&1

test -f "$root/atlas/manifest.tsv"
test -f "$results/phase2_2la.json"
test -f "$results/phase2_map_qc.json"
test -f "$results/phase2_resistance.json"
test -f "$results/phase2_pyrho.json"
test -f "$results/pedigree/phase2_pedigree.json"
find "$root/atlas" "$results" -type f -print0 | sort -z | xargs -0 shasum -a 256 \
  > "$root/provenance/postprocess_files.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$root/provenance/postprocess_completed_utc.txt"
