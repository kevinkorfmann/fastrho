#!/usr/bin/env bash
set -euo pipefail

: "${FASTRHO_REPO:?submit.sh must export FASTRHO_REPO}"
: "${MODEL_ROOT:?submit.sh must export MODEL_ROOT}"
: "${FASTRHO_PYTHON:?submit.sh must export FASTRHO_PYTHON}"

if [[ ! -x "${FASTRHO_PYTHON}" ]]; then
  echo "FASTRHO_PYTHON is not executable: ${FASTRHO_PYTHON}" >&2
  exit 2
fi
if [[ ! -f "${FASTRHO_REPO}/models/domain-randomized-v1/reproduce/workflow.json" ]]; then
  echo "FASTRHO_REPO does not contain the frozen workflow" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="${FASTRHO_REPO}"
mkdir -p "${MODEL_ROOT}/logs" "${MODEL_ROOT}/release"
