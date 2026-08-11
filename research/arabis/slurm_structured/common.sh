#!/usr/bin/env bash
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
export ARABIS_STRUCTURED_ROOT="${ARABIS_STRUCTURED_ROOT:-${ARABIS_ROOT}/structured_selfing_v1}"
export NUMBA_CACHE_DIR="${ARABIS_NODE_TMP}/numba"
mkdir -p "${ARABIS_STRUCTURED_ROOT}" "${ARABIS_ROOT}/logs" "${NUMBA_CACHE_DIR}"
