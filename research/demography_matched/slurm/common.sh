#!/usr/bin/env bash

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Demography-matched benchmark stages must run inside a Betty Slurm allocation" >&2
  return 2 2>/dev/null || exit 2
fi

export DEMO_REPO="${DEMO_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
export DEMO_ROOT="${DEMO_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-demography-v1}"
export DEMO_RELERNNSRC="${DEMO_ROOT}/software/ReLERNN-6655efd-weights"
export DEMO_RELERNN_PATCH="${DEMO_REPO}/research/demography_matched/patches/relernn_keras3_hdf5.patch"
export DEMO_RELERNN_IMAGE="${DEMO_ROOT}/software/nvidia-tensorflow-25.02-tf2-py3.sif"
export DEMO_RELERNN_ENV="${DEMO_ROOT}/envs/relernn-ngc"
export DEMO_PYRHO_ENV="${DEMO_ROOT}/envs/pyrho-py312"
export DEMO_UV="${HOME}/.local/bin/uv"
export UV_CACHE_DIR="${DEMO_ROOT}/cache/uv"
export XDG_CACHE_HOME="${DEMO_ROOT}/cache/xdg"
export CUDA_CACHE_PATH="${DEMO_ROOT}/cache/cuda"
export APPTAINER_CACHEDIR="${DEMO_ROOT}/cache/apptainer"
export PYTHONPYCACHEPREFIX="${DEMO_ROOT}/cache/pycache"
export PYTHONPATH="${DEMO_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${DEMO_ROOT}/logs" "${DEMO_ROOT}/results" "${DEMO_ROOT}/runtime" "${DEMO_ROOT}/cache"
[[ -x "${DEMO_UV}" ]] || { echo "uv not found: ${DEMO_UV}" >&2; exit 2; }
cd "${DEMO_REPO}"

load_apptainer() {
  if ! command -v apptainer >/dev/null 2>&1; then
    if [[ -n "${MODULESHOME:-}" && -f "${MODULESHOME}/init/bash" ]]; then
      source "${MODULESHOME}/init/bash"
    else
      source /vast/parcc/spack/sw/apps/linux-sapphirerapids/lmod-8.7.55-3et5rja7d5nh2lls7yg3byz25ygcirkw/lmod/lmod/init/bash
    fi
    module load apptainer/1.4.1
  fi
}

relernn_run() {
  load_apptainer
  [[ -f "${DEMO_RELERNN_IMAGE}" ]] || { echo "ReLERNN image missing: ${DEMO_RELERNN_IMAGE}" >&2; return 2; }
  [[ -x "${DEMO_RELERNN_ENV}/bin/python" ]] || { echo "ReLERNN environment missing: ${DEMO_RELERNN_ENV}" >&2; return 2; }
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
    --bind "${DEMO_ROOT}:${DEMO_ROOT},${DEMO_REPO}:${DEMO_REPO}" \
    "${DEMO_RELERNN_IMAGE}" "${command}" "$@"
}
