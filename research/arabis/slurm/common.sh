#!/usr/bin/env bash

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Arabis benchmark stages must run inside a Betty Slurm allocation" >&2
  return 2 2>/dev/null || exit 2
fi

export ARABIS_REPO="${ARABIS_REPO:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is unset}}"
export ARABIS_ROOT="${ARABIS_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-arabis}"
export ARABIS_WORK="${ARABIS_WORK:-${ARABIS_ROOT}/data}"
export ARABIS_ENV="${ARABIS_ENV:-${ARABIS_ROOT}/python}"
export ARABIS_MODEL_ROOT="${ARABIS_MODEL_ROOT:-${ARABIS_ROOT}/model}"
export ARABIS_CHECKPOINT="${ARABIS_CHECKPOINT:-${ARABIS_MODEL_ROOT}/epoch=48-val_loss=0.109.ckpt}"
export ARABIS_STATS="${ARABIS_STATS:-${ARABIS_MODEL_ROOT}/feat_stats.npz}"
export ARABIS_PIXI="${ARABIS_PIXI:-${HOME}/.pixi/bin/pixi}"
export UV_CACHE_DIR="${ARABIS_ROOT}/cache/uv"
export XDG_CACHE_HOME="${ARABIS_ROOT}/cache/xdg"
export TRITON_CACHE_DIR="${ARABIS_ROOT}/cache/triton"
export PYTHONPYCACHEPREFIX="${ARABIS_ROOT}/cache/pycache"
export UV_PROJECT_ENVIRONMENT="${ARABIS_ENV}"
export ARABIS_NODE_TMP="${SLURM_TMPDIR:-/tmp/${USER}/arabis-${SLURM_JOB_ID}}"

mkdir -p \
  "${ARABIS_ROOT}" "${ARABIS_WORK}" "${ARABIS_MODEL_ROOT}" \
  "${ARABIS_ROOT}/logs" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" "${PYTHONPYCACHEPREFIX}" "${ARABIS_NODE_TMP}"

[[ -x "${ARABIS_PIXI}" ]] || { echo "pixi not found: ${ARABIS_PIXI}" >&2; exit 2; }
cd "${ARABIS_REPO}"

arabis_bio() {
  "${ARABIS_PIXI}" run --manifest-path \
    "${ARABIS_REPO}/research/arabis/pixi.toml" "$@"
}
