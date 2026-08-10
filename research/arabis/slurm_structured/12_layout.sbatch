#!/usr/bin/env bash
#SBATCH --job-name=struct-layout
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
train_count=$(find "${ARABIS_STRUCTURED_ROOT}/shards/train" -maxdepth 1 -name 'ts_*.npz' | wc -l)
val_count=$(find "${ARABIS_STRUCTURED_ROOT}/shards/val" -maxdepth 1 -name 'ts_*.npz' | wc -l)
audit_count=$(find "${ARABIS_STRUCTURED_ROOT}/shards/audit" -maxdepth 1 -name 'ts_*.npz' | wc -l)
[[ "${train_count}" -eq 16000 ]] || { echo "train shards=${train_count}, expected 16000" >&2; exit 1; }
[[ "${val_count}" -eq 1600 ]] || { echo "val shards=${val_count}, expected 1600" >&2; exit 1; }
[[ "${audit_count}" -eq 1600 ]] || { echo "audit shards=${audit_count}, expected 1600" >&2; exit 1; }
for seed in $(seq 0 6); do
  root="${ARABIS_STRUCTURED_ROOT}/seeds/seed${seed}"
  mkdir -p "${root}/shards" "${root}/logs"
  ln -sfn "${ARABIS_STRUCTURED_ROOT}/shards/train" "${root}/shards/train"
  ln -sfn "${ARABIS_STRUCTURED_ROOT}/shards/val" "${root}/shards/val"
done
rm -rf "${ARABIS_STRUCTURED_ROOT}/audit_strata"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_audit_layout.py" \
  --input-dir "${ARABIS_STRUCTURED_ROOT}/shards/audit" \
  --output-root "${ARABIS_STRUCTURED_ROOT}/audit_strata" --expected 1600
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_simulation_design_audit.py" \
  --shard-root "${ARABIS_STRUCTURED_ROOT}/shards" \
  --train 16000 --val 1600 --audit 1600 \
  --output "${ARABIS_STRUCTURED_ROOT}/simulation_design_audit.json"
"${ARABIS_ENV}/bin/python" - "${ARABIS_STRUCTURED_ROOT}/audit_strata/layout.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
expected = {"panmictic", "diffuse_island", "nemorensis_panel", "sagittata_panel"}
assert set(payload["strata"]) == expected
assert min(payload["strata"].values()) >= 200
PY
printf 'campaign\ttrain_shards\tval_shards\taudit_shards\nstructured_selfing_v1\t%s\t%s\t%s\n' \
  "${train_count}" "${val_count}" "${audit_count}" > "${ARABIS_STRUCTURED_ROOT}/dataset.tsv"
