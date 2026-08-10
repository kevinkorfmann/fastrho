#!/usr/bin/env bash
#SBATCH --job-name=demo-rel-ngc
#SBATCH --partition=genoa-std-mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"
load_apptainer

mkdir -p "$(dirname "${DEMO_RELERNN_IMAGE}")" "$(dirname "${DEMO_RELERNN_ENV}")"
if [[ ! -f "${DEMO_RELERNN_IMAGE}" ]]; then
  tmp="${DEMO_RELERNN_IMAGE}.partial"
  rm -f "${tmp}"
  apptainer pull "${tmp}" docker://nvcr.io/nvidia/tensorflow:25.02-tf2-py3
  mv "${tmp}" "${DEMO_RELERNN_IMAGE}"
fi

if [[ ! -d "${DEMO_RELERNNSRC}/.git" ]]; then
  mkdir -p "$(dirname "${DEMO_RELERNNSRC}")"
  git clone https://github.com/kr-colab/ReLERNN.git "${DEMO_RELERNNSRC}"
fi
git -C "${DEMO_RELERNNSRC}" fetch origin
git -C "${DEMO_RELERNNSRC}" checkout --detach 6655efd
if git -C "${DEMO_RELERNNSRC}" apply --reverse --check "${DEMO_RELERNN_PATCH}" 2>/dev/null; then
  echo "ReLERNN Keras 3 serialization patch already applied"
else
  git -C "${DEMO_RELERNNSRC}" apply --check "${DEMO_RELERNN_PATCH}"
  git -C "${DEMO_RELERNNSRC}" apply "${DEMO_RELERNN_PATCH}"
fi
mapfile -t changed_files < <(git -C "${DEMO_RELERNNSRC}" diff --name-only)
expected_files=(ReLERNN/ReLERNN_PREDICT ReLERNN/ReLERNN_TRAIN ReLERNN/helpers.py)
[[ "${changed_files[*]}" == "${expected_files[*]}" ]] || {
  printf 'Unexpected ReLERNN source changes: %s\n' "${changed_files[*]}" >&2
  exit 2
}

if [[ ! -x "${DEMO_RELERNN_ENV}/bin/python" || ! -x "${DEMO_RELERNN_ENV}/bin/pip" ]]; then
  bootstrap="${DEMO_ROOT}/software/virtualenv-bootstrap"
  rm -rf "${DEMO_RELERNN_ENV}"
  mkdir -p "${bootstrap}"
  apptainer exec --bind "${DEMO_ROOT}:${DEMO_ROOT}" "${DEMO_RELERNN_IMAGE}" \
    python -m pip install --no-cache-dir --target "${bootstrap}" "virtualenv==20.31.2"
  apptainer exec --bind "${DEMO_ROOT}:${DEMO_ROOT}" "${DEMO_RELERNN_IMAGE}" \
    env PYTHONPATH="${bootstrap}" python -m virtualenv --system-site-packages "${DEMO_RELERNN_ENV}"
fi
apptainer exec --bind "${DEMO_ROOT}:${DEMO_ROOT}" "${DEMO_RELERNN_IMAGE}" \
  "${DEMO_RELERNN_ENV}/bin/pip" install \
  "numpy==1.26.4" "scipy==1.12.0" "msprime==1.3.4" "tskit==0.6.4" \
  "scikit-learn>=1.4,<2" "scikit-allel==1.3.13" "dask==2024.11.2" \
  "h5py>=3.10,<4" "matplotlib==3.9.2"
apptainer exec --bind "${DEMO_ROOT}:${DEMO_ROOT}" "${DEMO_RELERNN_IMAGE}" \
  "${DEMO_RELERNN_ENV}/bin/pip" install --no-deps --editable "${DEMO_RELERNNSRC}"

relernn_run python - <<'PY'
from pathlib import Path
from ReLERNN.helpers import check_demHist, convert_demHist
import os

root = Path(os.environ["DEMO_REPO"])
for scenario, expected_events in (("bottleneck", 2), ("expansion", 1)):
    path = root / "research" / "demography_matched" / "demographies" / f"{scenario}_smcpp.csv"
    kind = check_demHist(path)
    assert kind == 2
    converted = convert_demHist(path, 20, 1, kind, 1.5e-8)
    assert len(converted["demographic_events"]) == expected_events
PY
relernn_run python "${DEMO_REPO}/scripts/capture_demography_runtime.py" \
  --mode relernn --image "${DEMO_RELERNN_IMAGE}" \
  --patch "${DEMO_RELERNN_PATCH}" \
  --output "${DEMO_ROOT}/runtime/relernn.json"
