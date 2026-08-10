#!/usr/bin/env bash
set -euo pipefail
export PATH=/cm/local/apps/slurm/24.11/bin:${PATH}
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
base="ALL,ARABIS_ROOT=/vast/projects/smathi/cohort/${USER}/fastrho-arabis,ARABIS_REPO=${PWD}"
plan=$(sbatch --parsable --export="${base}" research/arabis/slurm_diagnosis/21_plan.sbatch); plan=${plan%%;*}
filter=$(sbatch --parsable --dependency="afterok:${plan}" --array=0-23 --export="${base}" \
  research/arabis/slurm_diagnosis/22_filter.sbatch); filter=${filter%%;*}
infer=$(sbatch --parsable --dependency="afterok:${filter}" --array=0-119%60 --export="${base}" \
  research/arabis/slurm_diagnosis/23_infer.sbatch); infer=${infer%%;*}
ensemble=$(sbatch --parsable --dependency="afterok:${infer}" --array=0-23 --export="${base}" \
  research/arabis/slurm_diagnosis/24_ensemble.sbatch); ensemble=${ensemble%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:${ensemble}" --export="${base}" \
  research/arabis/slurm_diagnosis/25_evaluate.sbatch); evaluate=${evaluate%%;*}
printf 'plan=%s filter=%s infer=%s ensemble=%s evaluate=%s\n' \
  "${plan}" "${filter}" "${infer}" "${ensemble}" "${evaluate}"
