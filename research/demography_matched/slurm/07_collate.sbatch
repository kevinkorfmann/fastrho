#!/usr/bin/env bash
#SBATCH --job-name=demo-collate
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"
cp "${DEMO_REPO}/research/demography_matched/design.json" "${DEMO_ROOT}/design.json"
cp "${DEMO_REPO}/research/demography_matched/fastrho_reference.json" \
  "${DEMO_ROOT}/fastrho_reference.json"
"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/score_fastrho_reference.py" \
  --root "${DEMO_ROOT}" --manifest "${DEMO_ROOT}/fastrho_reference.json" \
  --output "${DEMO_ROOT}/results/fastrho_reference.json"
"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/capture_demography_runtime.py" \
  --mode pyrho --output "${DEMO_ROOT}/runtime/pyrho.json"
relernn_run python "${DEMO_REPO}/scripts/capture_demography_runtime.py" \
  --mode relernn --image "${DEMO_RELERNN_IMAGE}" \
  --patch "${DEMO_RELERNN_PATCH}" \
  --output "${DEMO_ROOT}/runtime/relernn.json"
relernn_run python "${DEMO_REPO}/scripts/collate_demography_benchmark.py" \
  --root "${DEMO_ROOT}" --design "${DEMO_ROOT}/design.json" \
  --fastrho-reference "${DEMO_ROOT}/results/fastrho_reference.json" \
  --output "${DEMO_ROOT}/results/paired_demography_results.json"
relernn_run python "${DEMO_REPO}/scripts/archive_demography_predictions.py" \
  --root "${DEMO_ROOT}" \
  --output "${DEMO_ROOT}/results/paired_demography_predictions.npz"
(
  cd "${DEMO_ROOT}"
  sha256sum design.json fastrho_reference.json input_manifest.json \
    reference/bottleneck_pred_fastrho.npz reference/expansion_pred_fastrho.npz \
    runtime/pyrho.json runtime/relernn.json results/fastrho_reference.json \
    results/paired_demography_results.json results/paired_demography_predictions.npz \
    > results/SHA256SUMS
)
