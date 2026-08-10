#!/usr/bin/env bash
#SBATCH --job-name=arabis-resample-infer
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
root="${ARABIS_ROOT}/diagnosis/resamples"
panel_index=$((SLURM_ARRAY_TASK_ID / 5))
seed=$((SLURM_ARRAY_TASK_ID % 5))
line=$(sed -n "$((panel_index + 2))p" "${root}/panels.tsv")
IFS=$'\t' read -r panel _ _ _ _ <<< "${line}"
[[ -n "${panel}" ]]
manifest="${ARABIS_SMALLN_ROOT}/frozen_ensemble.json"
readarray -t paths < <("${ARABIS_ENV}/bin/python" -c \
  'import json,sys; m=json.load(open(sys.argv[1]))["members"][int(sys.argv[2])]; print(m["checkpoint"]); print(m["stats"])' \
  "${manifest}" "${seed}")
out="${root}/maps/${panel}/seed${seed}"
mkdir -p "${out}"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_infer.py" \
  --workdir "${ARABIS_WORK}" --maps-dir "${out}" \
  --checkpoint "${paths[0]}" --stats "${paths[1]}" --device cuda:0 \
  --panel "${panel}=${root}/vcf/${panel}.selfer.vcf.gz"
