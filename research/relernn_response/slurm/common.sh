#!/usr/bin/env bash

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ReLERNN response stages must run inside a Betty Slurm allocation" >&2
  return 2 2>/dev/null || exit 2
fi

export RESPONSE_REPO="${RESPONSE_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
export RESPONSE_ROOT="${RESPONSE_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-relernn-response-20260831}"
export RESPONSE_SOURCE_ROOT="${RESPONSE_SOURCE_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-demography-v1}"
export DEMO_ROOT="${RESPONSE_SOURCE_ROOT}"
export DEMO_RELERNNSRC="${DEMO_ROOT}/software/ReLERNN-6655efd-weights"
export DEMO_RELERNN_PATCH="${RESPONSE_REPO}/research/demography_matched/patches/relernn_keras3_hdf5.patch"
export DEMO_RELERNN_IMAGE="${DEMO_ROOT}/software/nvidia-tensorflow-25.02-tf2-py3.sif"
export DEMO_RELERNN_ENV="${DEMO_ROOT}/envs/relernn-ngc"
export XDG_CACHE_HOME="${RESPONSE_ROOT}/cache/xdg"
export CUDA_CACHE_PATH="${RESPONSE_ROOT}/cache/cuda"
export APPTAINER_CACHEDIR="${DEMO_ROOT}/cache/apptainer"
export PYTHONPYCACHEPREFIX="${RESPONSE_ROOT}/cache/pycache"
export PYTHONPATH="${RESPONSE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${RESPONSE_ROOT}/logs" "${RESPONSE_ROOT}/results" "${RESPONSE_ROOT}/cache"
cd "${RESPONSE_REPO}"

load_apptainer() {
  if ! command -v apptainer >/dev/null 2>&1; then
    source /vast/parcc/spack/sw/apps/linux-sapphirerapids/lmod-8.7.55-3et5rja7d5nh2lls7yg3byz25ygcirkw/lmod/lmod/init/bash
    module load apptainer/1.4.1
  fi
}

relernn_run() {
  load_apptainer
  [[ -f "${DEMO_RELERNN_IMAGE}" ]] || { echo "Missing image: ${DEMO_RELERNN_IMAGE}" >&2; return 2; }
  [[ -x "${DEMO_RELERNN_ENV}/bin/python" ]] || { echo "Missing environment: ${DEMO_RELERNN_ENV}" >&2; return 2; }
  local command=$1
  shift
  if [[ "${command}" == "python" ]]; then
    command="${DEMO_RELERNN_ENV}/bin/python"
  else
    command="${DEMO_RELERNN_ENV}/bin/${command}"
  fi
  local gpu_args=()
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
    gpu_args+=(--nv)
  fi
  apptainer exec "${gpu_args[@]}" \
    --bind "${RESPONSE_ROOT}:${RESPONSE_ROOT},${RESPONSE_SOURCE_ROOT}:${RESPONSE_SOURCE_ROOT},${RESPONSE_REPO}:${RESPONSE_REPO}" \
    "${DEMO_RELERNN_IMAGE}" "${command}" "$@"
}
