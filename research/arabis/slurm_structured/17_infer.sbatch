#!/usr/bin/env bash
#SBATCH --job-name=struct-infer
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=28
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
seed=${SLURM_ARRAY_TASK_ID}
manifest="${ARABIS_STRUCTURED_ROOT}/frozen_ensemble.json"
readarray -t paths < <("${ARABIS_ENV}/bin/python" -c \
  'import json,sys; m=json.load(open(sys.argv[1]))["members"][int(sys.argv[2])]; print(m["checkpoint"]); print(m["stats"])' \
  "${manifest}" "${seed}")
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_infer.py" \
  --workdir "${ARABIS_WORK}" --maps-dir "${ARABIS_STRUCTURED_ROOT}/arabis_maps/seed${seed}" \
  --checkpoint "${paths[0]}" --stats "${paths[1]}" --device cuda:0
