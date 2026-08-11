#!/usr/bin/env bash
set -euo pipefail

root="${1:-data/ag1000g-phase2-ar1}"
repo="${2:-$(git rev-parse --show-toplevel)}"
system_python="${FASTRHO_PHASE2_SYSTEM_PYTHON:-python3}"
fastrho_python="${FASTRHO_PHASE2_INFERENCE_PYTHON:-python3}"
checkpoint="${FASTRHO_PHASE2_CHECKPOINT:-$repo/downloaded-models/high-ne-v1/model.ckpt}"
stats="${FASTRHO_PHASE2_STATS:-$repo/downloaded-models/high-ne-v1/feat_stats.npz}"
pyrho="${FASTRHO_PHASE2_PYRHO:-pyrho}"
metadata="$root/raw/metadata/samples.meta.txt"
selection="${FASTRHO_PHASE2_SELECTION:-$repo/paper/anopheles_variants/phase2/cohorts/selection.tsv}"
selected="${FASTRHO_PHASE2_SELECTED_SAMPLES:-$repo/paper/anopheles_variants/phase2/cohorts/selected_samples.tsv}"
tag_snps="${FASTRHO_PHASE2_TAG_SNPS:-$repo/paper/anopheles_variants/phase2/provenance/karyotype_tag_snps.csv}"
panel_spec="${FASTRHO_PHASE2_PANEL_SPEC:-$repo/paper/anopheles_variants/phase2/provenance/anopheles_resistance_panels.tsv}"
maps="$root/maps"
normalized="$root/normalized"
results="$root/results"
logs="$root/logs"
source="${FASTRHO_PHASE2_SOURCE_DIR:-$repo/paper/anopheles_variants/common}"

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
  --tag-snps "$tag_snps" --maps "$maps" \
  --out "$results/phase2_2la.json" \
  > "$logs/phase2_2la.log" 2>&1 &
pids+=("$!")

"$system_python" "$source/phase2_resistance.py" \
  --normalized "$normalized" --maps "$maps" --selection "$selection" \
  --panel-spec "$panel_spec" \
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
