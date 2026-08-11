#!/usr/bin/env bash
set -euo pipefail
export PATH=/cm/local/apps/slurm/24.11/bin:${PATH}
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
base="ALL,ARABIS_ROOT=/vast/projects/smathi/cohort/${USER}/fastrho-arabis,ARABIS_REPO=${PWD}"
filter=$(sbatch --parsable --export="${base}" research/arabis/slurm_diagnosis/31_composition_filter.sbatch); filter=${filter%%;*}
infer=$(sbatch --parsable --dependency="afterok:${filter}" --array=0-14 --export="${base}" \
  research/arabis/slurm_diagnosis/32_composition_infer.sbatch); infer=${infer%%;*}
ensemble=$(sbatch --parsable --dependency="afterok:${infer}" --array=0-2 --export="${base}" \
  research/arabis/slurm_diagnosis/33_composition_ensemble.sbatch); ensemble=${ensemble%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:${ensemble}" --export="${base}" \
  research/arabis/slurm_diagnosis/34_composition_evaluate.sbatch); evaluate=${evaluate%%;*}
printf 'filter=%s infer=%s ensemble=%s evaluate=%s\n' \
  "${filter}" "${infer}" "${ensemble}" "${evaluate}"
