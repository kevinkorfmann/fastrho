#!/usr/bin/env bash
set -euo pipefail
SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/24.11/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${RESPONSE_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-relernn-response-20260831}
SOURCE_ROOT=${RESPONSE_SOURCE_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-demography-v1}
[[ ! -e "${ROOT}/arms" ]] || { echo "Refusing to reuse experiment root: ${ROOT}" >&2; exit 2; }
[[ -x "${SOURCE_ROOT}/envs/relernn-ngc/bin/python" ]] || { echo "Missing frozen ReLERNN environment" >&2; exit 2; }
mkdir -p "${ROOT}/logs" "${ROOT}/results"
cd "${REPO}"
base="ALL,RESPONSE_REPO=${REPO},RESPONSE_ROOT=${ROOT},RESPONSE_SOURCE_ROOT=${SOURCE_ROOT}"

prepare=$(sbatch --parsable --export="${base}" research/relernn_response/slurm/00_prepare.sbatch); prepare=${prepare%%;*}
simulate=$(sbatch --parsable --dependency="afterok:${prepare}" --array=0-2%3 --export="${base}" research/relernn_response/slurm/01_simulate.sbatch); simulate=${simulate%%;*}
train=$(sbatch --parsable --dependency="aftercorr:${simulate}" --array=0-2%3 --export="${base}" research/relernn_response/slurm/02_train_predict.sbatch); train=${train%%;*}
score=$(sbatch --parsable --dependency="afterok:${train}" --export="${base}" research/relernn_response/slurm/03_score.sbatch); score=${score%%;*}

printf 'prepare=%s\nsimulate=%s\ntrain_predict=%s\nscore=%s\n' \
  "${prepare}" "${simulate}" "${train}" "${score}" | tee "${ROOT}/submission.tsv"
