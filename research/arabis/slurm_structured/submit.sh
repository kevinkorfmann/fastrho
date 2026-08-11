#!/usr/bin/env bash
set -euo pipefail
SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/24.11/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${ARABIS_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-arabis}
STRUCTURED=${ARABIS_STRUCTURED_ROOT:-${ROOT}/structured_selfing_v1}
mkdir -p "${ROOT}/logs" "${STRUCTURED}"
cd "${REPO}"
base="ALL,ARABIS_REPO=${REPO},ARABIS_ROOT=${ROOT},ARABIS_STRUCTURED_ROOT=${STRUCTURED}"

env_job=$(sbatch --parsable --export="${base}" research/arabis/slurm/00_inference_env.sbatch); env_job=${env_job%%;*}
preregister=$(sbatch --parsable --dependency="afterok:${env_job}" --export="${base}" \
  research/arabis/slurm_structured/08_preregister.sbatch); preregister=${preregister%%;*}
smoke=$(sbatch --parsable --dependency="afterok:${preregister}" --export="${base}" \
  research/arabis/slurm_structured/09_smoke.sbatch); smoke=${smoke%%;*}

train_gen=$(sbatch --parsable --dependency="afterok:${smoke}" --array=0-63%64 \
  --export="${base},STRUCTURED_SPLIT=train,STRUCTURED_PER_TASK=250,STRUCTURED_OFFSET_BASE=0" \
  research/arabis/slurm_structured/10_generate.sbatch); train_gen=${train_gen%%;*}
val_gen=$(sbatch --parsable --dependency="afterok:${smoke}" --array=0-15%16 \
  --export="${base},STRUCTURED_SPLIT=val,STRUCTURED_PER_TASK=100,STRUCTURED_OFFSET_BASE=1000000" \
  research/arabis/slurm_structured/10_generate.sbatch); val_gen=${val_gen%%;*}
audit_gen=$(sbatch --parsable --dependency="afterok:${smoke}" --array=0-15%16 \
  --export="${base},STRUCTURED_SPLIT=audit,STRUCTURED_PER_TASK=100,STRUCTURED_OFFSET_BASE=2000000" \
  research/arabis/slurm_structured/10_generate.sbatch); audit_gen=${audit_gen%%;*}

train_feat=$(sbatch --parsable --dependency="aftercorr:${train_gen}" --array=0-63%64 \
  --export="${base},STRUCTURED_SPLIT=train" research/arabis/slurm_structured/11_preprocess.sbatch); train_feat=${train_feat%%;*}
val_feat=$(sbatch --parsable --dependency="aftercorr:${val_gen}" --array=0-15%16 \
  --export="${base},STRUCTURED_SPLIT=val" research/arabis/slurm_structured/11_preprocess.sbatch); val_feat=${val_feat%%;*}
audit_feat=$(sbatch --parsable --dependency="aftercorr:${audit_gen}" --array=0-15%16 \
  --export="${base},STRUCTURED_SPLIT=audit" research/arabis/slurm_structured/11_preprocess.sbatch); audit_feat=${audit_feat%%;*}

layout=$(sbatch --parsable --dependency="afterok:${train_feat}:${val_feat}:${audit_feat}" --export="${base}" \
  research/arabis/slurm_structured/12_layout.sbatch); layout=${layout%%;*}
train=$(sbatch --parsable --dependency="afterok:${layout}" --array=0-6%7 --export="${base}" \
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

printf 'environment=%s\npreregister=%s\nsmoke=%s\ntrain_generate=%s\nval_generate=%s\naudit_generate=%s\ntrain_features=%s\nval_features=%s\naudit_features=%s\nlayout=%s\ntrain=%s\nfreeze=%s\nsimulation_audit=%s\ngate=%s\ninfer=%s\nensemble=%s\nevaluate=%s\nfinalize=%s\n' \
  "${env_job}" "${preregister}" "${smoke}" "${train_gen}" "${val_gen}" "${audit_gen}" \
  "${train_feat}" "${val_feat}" "${audit_feat}" "${layout}" "${train}" "${freeze}" \
  "${audit}" "${gate}" "${infer}" "${ensemble}" "${evaluate}" "${finalize}" | tee "${STRUCTURED}/submission.tsv"
