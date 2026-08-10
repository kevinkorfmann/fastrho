#!/usr/bin/env bash
set -euo pipefail
export PATH=/cm/local/apps/slurm/24.11/bin:${PATH}
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
base="ALL,ARABIS_ROOT=/vast/projects/smathi/cohort/${USER}/fastrho-arabis,ARABIS_REPO=${PWD}"
infer=$(sbatch --parsable --array=0-4 --export="${base}" \
  research/arabis/slurm_diagnosis/35_condition2_infer.sbatch); infer=${infer%%;*}
ensemble=$(sbatch --parsable --dependency="afterok:${infer}" --export="${base}" \
  research/arabis/slurm_diagnosis/36_condition2_ensemble.sbatch); ensemble=${ensemble%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:${ensemble}" --export="${base}" \
  research/arabis/slurm_diagnosis/37_condition2_evaluate.sbatch); evaluate=${evaluate%%;*}
printf 'infer=%s ensemble=%s evaluate=%s\n' "${infer}" "${ensemble}" "${evaluate}"
