#!/usr/bin/env bash
set -euo pipefail
SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/24.11/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${ARABIS_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-arabis}
STRUCTURED=${ARABIS_STRUCTURED_ROOT:-${ROOT}/structured_selfing_v1}
: "${OLD_TRAIN_JOB:?set OLD_TRAIN_JOB}" "${OLD_TAIL_JOBS:?set OLD_TAIL_JOBS}"
cd "${REPO}"
base="ALL,ARABIS_REPO=${REPO},ARABIS_ROOT=${ROOT},ARABIS_STRUCTURED_ROOT=${STRUCTURED}"

scancel "${OLD_TRAIN_JOB}" ${OLD_TAIL_JOBS//,/ }
cleanup=$(sbatch --parsable --dependency="afterany:${OLD_TRAIN_JOB}" --export="${base}" \
  research/arabis/slurm_structured/12b_cleanup_training.sbatch); cleanup=${cleanup%%;*}
train=$(sbatch --parsable --dependency="afterok:${cleanup}" --array=0-6%7 --export="${base}" \
  research/arabis/slurm_structured/13_train.sbatch); train=${train%%;*}
freeze=$(sbatch --parsable --dependency="afterok:${train}" --export="${base}" \
  research/arabis/slurm_structured/14_freeze.sbatch); freeze=${freeze%%;*}
audit=$(sbatch --parsable --dependency="afterok:${freeze}" --array=0-6%7 --export="${base}" \
  research/arabis/slurm_structured/15_simulation_audit.sbatch); audit=${audit%%;*}
gate=$(sbatch --parsable --dependency="afterok:${audit}" --export="${base}" \
  research/arabis/slurm_structured/16_gate.sbatch); gate=${gate%%;*}
infer=$(sbatch --parsable --dependency="afterok:${gate}" --array=0-6%7 --export="${base}" \
  research/arabis/slurm_structured/17_infer.sbatch); infer=${infer%%;*}
ensemble=$(sbatch --parsable --dependency="afterok:${infer}" --export="${base}" \
  research/arabis/slurm_structured/18_ensemble.sbatch); ensemble=${ensemble%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:${ensemble}" --export="${base}" \
  research/arabis/slurm_structured/19_evaluate.sbatch); evaluate=${evaluate%%;*}
finalize=$(sbatch --parsable --dependency="afterok:${evaluate}" --export="${base}" \
  research/arabis/slurm_structured/20_finalize.sbatch); finalize=${finalize%%;*}

printf 'hardware=full_B200\nold_train=%s\ncleanup=%s\ntrain=%s\nfreeze=%s\nsimulation_audit=%s\ngate=%s\ninfer=%s\nensemble=%s\nevaluate=%s\nfinalize=%s\n' \
  "${OLD_TRAIN_JOB}" "${cleanup}" "${train}" "${freeze}" "${audit}" "${gate}" \
  "${infer}" "${ensemble}" "${evaluate}" "${finalize}" \
  | tee "${STRUCTURED}/resubmission_full_b200.tsv"
