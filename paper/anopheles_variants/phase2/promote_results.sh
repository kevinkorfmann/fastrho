#!/usr/bin/env bash
set -euo pipefail

remote_host="${1:-sesame}"
remote_root="${2:-/home/kkor/agam_phase2}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
common="$repo_root/paper/anopheles_variants/common"
plot_python="${PHASE2_PLOT_PYTHON:-python3}"

remote_count="$({ ssh "$remote_host" \
  "find '$remote_root/maps' -maxdepth 1 -type f -name '*.npz' | wc -l"; } | tr -d '[:space:]')"
if [[ "$remote_count" != 45 ]]; then
  echo "Refusing promotion: expected 45 remote maps, found $remote_count" >&2
  exit 1
fi
ssh "$remote_host" "test -f '$remote_root/provenance/postprocess_completed_utc.txt' \
  && test -f '$remote_root/results/phase2_map_qc.json' \
  && test -f '$remote_root/results/phase2_2la.json' \
  && test -f '$remote_root/results/phase2_resistance.json' \
  && test -f '$remote_root/results/phase2_pyrho.json' \
  && test -f '$remote_root/results/pedigree/phase2_pedigree.json' \
  && test -f '$remote_root/atlas/manifest.tsv'"

mkdir -p \
  "$script_dir/maps" \
  "$script_dir/results" \
  "$script_dir/release/atlas_anopheles" \
  "$script_dir/provenance/remote" \
  "$script_dir/figures" \
  "$script_dir/figdata" \
  "$script_dir/tables" \
  "$script_dir/manuscript/generated" \
  "$script_dir/manuscript/figures"

rsync -a --delete "$remote_host:$remote_root/maps/" "$script_dir/maps/"
rsync -a --delete "$remote_host:$remote_root/results/" "$script_dir/results/"
rsync -a --delete "$remote_host:$remote_root/atlas/" "$script_dir/release/atlas_anopheles/"
rsync -a \
  "$remote_host:$remote_root/provenance/normalized_files.sha256" \
  "$remote_host:$remote_root/provenance/map_files.sha256" \
  "$remote_host:$remote_root/provenance/postprocess_files.sha256" \
  "$remote_host:$remote_root/provenance/atlas_completed_utc.txt" \
  "$remote_host:$remote_root/provenance/postprocess_completed_utc.txt" \
  "$remote_host:$remote_root/provenance/compute_environment.txt" \
  "$remote_host:$remote_root/provenance/fastrho_environment.freeze.txt" \
  "$remote_host:$remote_root/provenance/pyrho_environment.freeze.txt" \
  "$remote_host:$remote_root/provenance/system_environment.freeze.txt" \
  "$remote_host:$remote_root/provenance/model_files.sha256" \
  "$remote_host:$remote_root/provenance/remote_source_state.txt" \
  "$script_dir/provenance/remote/"

# Keep tracked text artifacts platform-independent even when a remote producer
# writes CRLF. Normalize before figures, generated prose, and provenance hashes
# are rebuilt so a promotion is reproducible on macOS, Linux, and GitHub.
"$plot_python" - \
  "$script_dir/release/atlas_anopheles/manifest.tsv" \
  "$script_dir/results/pedigree/phase2_pedigree_windows.tsv" <<'PY'
from pathlib import Path
import sys

for filename in sys.argv[1:]:
    output = Path(filename)
    payload = output.read_bytes()
    output.write_bytes(payload.replace(b"\r\n", b"\n"))
PY

"$plot_python" "$common/plot_phase2.py" \
  --maps "$script_dir/maps" \
  --selection "$script_dir/cohorts/selection.tsv" \
  --results "$script_dir/results" \
  --out "$script_dir/figures"

"$plot_python" "$common/phase2_manuscript_data.py" \
  --results "$script_dir/results" \
  --selection "$script_dir/cohorts/selection.tsv" \
  --out "$script_dir/results/manuscript_generated"

rsync -a "$script_dir/results/manuscript_generated/" "$script_dir/manuscript/generated/"
rsync -a "$script_dir/figures/" "$script_dir/manuscript/figures/"
rsync -a "$script_dir/results/phase2_pyrho.npz" "$script_dir/figdata/"
rsync -a "$script_dir/results/phase2_2la.json" "$script_dir/figdata/"
rsync -a "$script_dir/results/phase2_resistance.json" "$script_dir/figdata/"
rsync -a "$script_dir/results/pedigree/phase2_pedigree.json" "$script_dir/figdata/"
rsync -a "$script_dir/results/manuscript_generated/phase2_cohorts.tex" "$script_dir/tables/"

(
  cd "$repo_root"
  find \
    paper/anopheles_variants/common \
    paper/anopheles_variants/phase2/fragments \
    paper/anopheles_variants/phase2/cohorts \
    -type f \( -name '*.py' -o -name '*.tex' -o -name '*.tsv' -o -name '*.json' \) -print0 \
    | sort -z | xargs -0 shasum -a 256
  find paper/anopheles_variants/phase2 -maxdepth 1 -type f \
    \( -name '*.sh' -o -name '*.json' -o -name '*.md' \) -print0 \
    | sort -z | xargs -0 shasum -a 256
) > "$script_dir/provenance/source_code.sha256"

(
  cd "$script_dir"
  find maps results release figures figdata tables manuscript \
    -type f -print0 | sort -z | xargs -0 shasum -a 256
) > "$script_dir/provenance/promoted_files.sha256"

echo "Phase 2 artifacts promoted. Inspect figures and run the submission audit before setting config.json complete."
