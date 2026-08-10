#!/usr/bin/env bash
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
export ARABIS_SMALLN_ROOT="${ARABIS_SMALLN_ROOT:-${ARABIS_ROOT}/smalln_self3}"
export NUMBA_CACHE_DIR="${ARABIS_NODE_TMP}/numba"
mkdir -p "${ARABIS_SMALLN_ROOT}" "${ARABIS_SMALLN_ROOT}/logs" "${NUMBA_CACHE_DIR}"
