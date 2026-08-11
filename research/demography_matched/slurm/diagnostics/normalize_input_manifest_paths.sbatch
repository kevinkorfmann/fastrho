#!/usr/bin/env bash
#SBATCH --job-name=demo-portable-manifest
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"

relernn_run python - "${DEMO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
path = root / "input_manifest.json"
manifest = json.loads(path.read_text())
for record in manifest["scenarios"].values():
    for key in ("raw_source", "validated_source"):
        source = Path(record[key])
        if source.is_absolute():
            record[key] = str(source.resolve().relative_to(root))
path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
