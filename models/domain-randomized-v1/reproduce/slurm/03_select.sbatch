#!/usr/bin/env bash
#SBATCH --job-name=fastrho-dr-select
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/models/domain-randomized-v1/reproduce/slurm/common.sh"

mapfile -t metrics_files < <(find "${MODEL_ROOT}/training" -type f -name metrics.csv | sort)
if (( ${#metrics_files[@]} != 1 )); then
  echo "expected exactly one metrics.csv, found ${#metrics_files[@]}" >&2
  exit 2
fi
metrics=${metrics_files[0]}
version_dir=$(dirname "${metrics}")
stats="${MODEL_ROOT}/shards/feat_stats_dr.npz"

"${FASTRHO_PYTHON}" "${FASTRHO_REPO}/scripts/select_model_checkpoint.py" \
  --model-id domain-randomized-v1 \
  --metrics "${metrics}" --checkpoint-dir "${version_dir}/checkpoints" \
  --stats "${stats}" --metric val_pearson --mode max \
  --output "${MODEL_ROOT}/release/reproduction-selection.json"

"${FASTRHO_PYTHON}" - "${MODEL_ROOT}" "${FASTRHO_REPO}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
repo = pathlib.Path(sys.argv[2])
record = root / "release" / "reproduction-selection.json"
workflow = repo / "models" / "domain-randomized-v1" / "reproduce" / "workflow.json"
payload = json.loads(record.read_text())
payload["workflow_sha256"] = hashlib.sha256(workflow.read_bytes()).hexdigest()
record.write_text(json.dumps(payload, indent=2) + "\n")
files = [record, workflow, root / "provenance" / "python.txt",
         root / "provenance" / "python-packages.txt",
         root / "provenance" / "repository-commit.txt",
         root / "provenance" / "repository-status.txt",
         root / "provenance" / "nvidia-smi.txt"]
lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files]
(root / "release" / "SHA256SUMS").write_text("".join(lines))
PY
