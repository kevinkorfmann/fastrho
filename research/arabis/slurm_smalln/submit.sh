#!/usr/bin/env bash
set -euo pipefail
SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/24.11/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${ARABIS_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-arabis}
SMALLN=${ARABIS_SMALLN_ROOT:-${ROOT}/smalln_self3}
mkdir -p "${ROOT}/logs" "${SMALLN}"
cd "${REPO}"
base="ALL,ARABIS_REPO=${REPO},ARABIS_ROOT=${ROOT},ARABIS_SMALLN_ROOT=${SMALLN}"

env_job=$(sbatch --parsable --export="${base}" research/arabis/slurm/00_inference_env.sbatch); env_job=${env_job%%;*}
train_gen=$(sbatch --parsable --dependency="afterok:${env_job}" --array=0-31%32 \
  --export="${base},SMALLN_SPLIT=train,SMALLN_PER_TASK=250,SMALLN_OFFSET_BASE=0" \
  research/arabis/slurm_smalln/10_generate.sbatch); train_gen=${train_gen%%;*}
val_gen=$(sbatch --parsable --dependency="afterok:${env_job}" --array=0-7%8 \
  --export="${base},SMALLN_SPLIT=val,SMALLN_PER_TASK=100,SMALLN_OFFSET_BASE=1000000" \
  research/arabis/slurm_smalln/10_generate.sbatch); val_gen=${val_gen%%;*}
train_feat=$(sbatch --parsable --dependency="aftercorr:${train_gen}" --array=0-31%32 \
  --export="${base},SMALLN_SPLIT=train" research/arabis/slurm_smalln/11_preprocess.sbatch); train_feat=${train_feat%%;*}
val_feat=$(sbatch --parsable --dependency="aftercorr:${val_gen}" --array=0-7%8 \
  --export="${base},SMALLN_SPLIT=val" research/arabis/slurm_smalln/11_preprocess.sbatch); val_feat=${val_feat%%;*}
layout=$(sbatch --parsable --dependency="afterok:${train_feat}:${val_feat}" --export="${base}" \
  research/arabis/slurm_smalln/12_layout.sbatch); layout=${layout%%;*}
train=$(sbatch --parsable --dependency="afterok:${layout}" --array=0-4%5 --export="${base}" \
  research/arabis/slurm_smalln/13_train.sbatch); train=${train%%;*}
freeze=$(sbatch --parsable --dependency="afterok:${train}" --export="${base}" \
  research/arabis/slurm_smalln/14_freeze.sbatch); freeze=${freeze%%;*}
infer=$(sbatch --parsable --dependency="afterok:${freeze}" --array=0-4%5 --export="${base}" \
  research/arabis/slurm_smalln/15_infer.sbatch); infer=${infer%%;*}
ensemble=$(sbatch --parsable --dependency="afterok:${infer}" --export="${base}" \
  research/arabis/slurm_smalln/16_ensemble.sbatch); ensemble=${ensemble%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:${ensemble}" --export="${base}" \
  research/arabis/slurm_smalln/17_evaluate.sbatch); evaluate=${evaluate%%;*}

printf 'environment=%s\ntrain_generate=%s\nval_generate=%s\ntrain_features=%s\nval_features=%s\nlayout=%s\ntrain=%s\nfreeze=%s\ninfer=%s\nensemble=%s\nevaluate=%s\n' \
  "${env_job}" "${train_gen}" "${val_gen}" "${train_feat}" "${val_feat}" "${layout}" \
  "${train}" "${freeze}" "${infer}" "${ensemble}" "${evaluate}" | tee "${SMALLN}/submission.tsv"
