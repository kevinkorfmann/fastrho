#!/usr/bin/env bash
set -euo pipefail
SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/current/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${DEMO_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-demography-v1}
mkdir -p "${ROOT}/logs" "${ROOT}/results"
cd "${REPO}"
base="ALL,DEMO_REPO=${REPO},DEMO_ROOT=${ROOT}"

pyrho_setup=$(sbatch --parsable --export="${base}" research/demography_matched/slurm/00_pyrho_environment.sbatch); pyrho_setup=${pyrho_setup%%;*}
relernn_setup=$(sbatch --parsable --export="${base}" research/demography_matched/slurm/01_relernn_environment.sbatch); relernn_setup=${relernn_setup%%;*}
prepare=$(sbatch --parsable --dependency="afterok:${relernn_setup}" --export="${base}" research/demography_matched/slurm/02_prepare.sbatch); prepare=${prepare%%;*}
smoke=$(sbatch --parsable --dependency="afterok:${relernn_setup}" --export="${base}" research/demography_matched/slurm/03_relernn_smoke.sbatch); smoke=${smoke%%;*}
rel_sim=$(sbatch --parsable --dependency="afterok:${prepare}:${smoke}" --array=0-3%4 --export="${base}" research/demography_matched/slurm/04_relernn_simulate.sbatch); rel_sim=${rel_sim%%;*}
rel_train=$(sbatch --parsable --dependency="aftercorr:${rel_sim}" --array=0-3%4 --export="${base}" research/demography_matched/slurm/05_relernn_train_score.sbatch); rel_train=${rel_train%%;*}
pyrho=$(sbatch --parsable --dependency="afterok:${prepare}:${pyrho_setup}" --array=0-3%4 --export="${base}" research/demography_matched/slurm/06_pyrho_infer_score.sbatch); pyrho=${pyrho%%;*}
collate=$(sbatch --parsable --dependency="afterok:${rel_train}:${pyrho}" --export="${base}" research/demography_matched/slurm/07_collate.sbatch); collate=${collate%%;*}

printf 'pyrho_environment=%s\nrelernn_environment=%s\nprepare=%s\nsmoke=%s\nrelernn_simulate=%s\nrelernn_train_score=%s\npyrho=%s\ncollate=%s\n' \
  "${pyrho_setup}" "${relernn_setup}" "${prepare}" "${smoke}" "${rel_sim}" "${rel_train}" "${pyrho}" "${collate}" \
  | tee "${ROOT}/submission.tsv"
