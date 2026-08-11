#!/usr/bin/env bash
#SBATCH --job-name=struct-smoke
#SBATCH --partition=genoa-std-mem
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
rm -rf "${ARABIS_STRUCTURED_ROOT}/smoke"
mkdir -p "${ARABIS_STRUCTURED_ROOT}/smoke/sims" "${ARABIS_STRUCTURED_ROOT}/smoke/shards"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_structured_selfing_gen.py" \
  --out-dir "${ARABIS_STRUCTURED_ROOT}/smoke/sims" --count 24 --offset 9000000 \
  --workers "${SLURM_CPUS_PER_TASK}"
"${ARABIS_ENV}/bin/python" -m fastrho.preprocess \
  --sim-dir "${ARABIS_STRUCTURED_ROOT}/smoke/sims" \
  --out-dir "${ARABIS_STRUCTURED_ROOT}/smoke/shards" \
  --with-features --num-processes "${SLURM_CPUS_PER_TASK}"
"${ARABIS_ENV}/bin/python" - "${ARABIS_STRUCTURED_ROOT}/smoke" <<'PY'
import json
import pathlib
import sys
import numpy as np

root = pathlib.Path(sys.argv[1])
metadata = []
for path in sorted((root / "shards").glob("ts_*.npz")):
    with np.load(path, allow_pickle=True) as archive:
        assert archive["tokens"].shape[0] >= 2
        meta = json.loads(str(archive["meta"]))
        assert meta["n_haplotypes"] == sum(meta["sample_counts_by_deme"])
        assert not meta["cross_map_used"]
        metadata.append(meta)
assert len(metadata) == 24
assert len({row["design"] for row in metadata}) >= 3
summary = {
    "n": len(metadata),
    "designs": sorted({row["design"] for row in metadata}),
    "n_haplotypes": sorted({row["n_haplotypes"] for row in metadata}),
    "selfing_range": [min(row["selfing"] for row in metadata), max(row["selfing"] for row in metadata)],
    "site_range": [min(row["num_sites"] for row in metadata), max(row["num_sites"] for row in metadata)],
}
(root / "smoke.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
