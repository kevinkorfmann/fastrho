#!/usr/bin/env bash
set -euo pipefail

SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/current/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${MODEL_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-model-reproduction}
: "${FASTRHO_PYTHON:?set FASTRHO_PYTHON to the CUDA environment python executable}"

if [[ -e "${ROOT}/submission.tsv" || -e "${ROOT}/simulations" || -e "${ROOT}/training" ]]; then
  echo "MODEL_ROOT already contains a run; choose a fresh directory: ${ROOT}" >&2
  exit 2
fi
mkdir -p "${ROOT}/logs"
cd "${REPO}"
base="ALL,FASTRHO_REPO=${REPO},MODEL_ROOT=${ROOT},FASTRHO_PYTHON=${FASTRHO_PYTHON}"

simulate=$(sbatch --parsable --array=0-63%32 \
  --output="${ROOT}/logs/%x-%A_%a.out" --error="${ROOT}/logs/%x-%A_%a.err" --export="${base}" \
  models/domain-randomized-v1/reproduce/slurm/00_simulate.sbatch); simulate=${simulate%%;*}
preprocess=$(sbatch --parsable --dependency="afterok:${simulate}" --array=0-5%6 \
  --output="${ROOT}/logs/%x-%A_%a.out" --error="${ROOT}/logs/%x-%A_%a.err" --export="${base}" \
  models/domain-randomized-v1/reproduce/slurm/01_preprocess.sbatch); preprocess=${preprocess%%;*}
train=$(sbatch --parsable --dependency="afterok:${preprocess}" \
  --output="${ROOT}/logs/%x-%j.out" --error="${ROOT}/logs/%x-%j.err" --export="${base}" \
  models/domain-randomized-v1/reproduce/slurm/02_train.sbatch); train=${train%%;*}
select=$(sbatch --parsable --dependency="afterok:${train}" \
  --output="${ROOT}/logs/%x-%j.out" --error="${ROOT}/logs/%x-%j.err" --export="${base}" \
  models/domain-randomized-v1/reproduce/slurm/03_select.sbatch); select=${select%%;*}

printf 'simulate=%s\npreprocess=%s\ntrain=%s\nselect=%s\n' \
  "${simulate}" "${preprocess}" "${train}" "${select}" | tee "${ROOT}/submission.tsv"
