#!/usr/bin/env bash
#SBATCH --job-name=demo-prepare
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"
input_archive="${DEMO_REPO}/docs/data/downloads/demography_matched_inputs.zip"
if [[ ! -f "${DEMO_ROOT}/input/bottleneck_n20/config.json" && \
      ! -f "${DEMO_ROOT}/input/expansion_n20/config.json" ]]; then
  mkdir -p "${DEMO_ROOT}/input"
  unzip -q "${input_archive}" -d "${DEMO_ROOT}/input"
fi
for scenario in bottleneck_n20 expansion_n20; do
  [[ -f "${DEMO_ROOT}/input/${scenario}/config.json" ]] || {
    echo "Incomplete benchmark input: ${DEMO_ROOT}/input/${scenario}" >&2
    exit 2
  }
done
relernn_run python "${DEMO_REPO}/scripts/prepare_demography_benchmark.py" --root "${DEMO_ROOT}"
cp "${DEMO_REPO}/research/demography_matched/design.json" "${DEMO_ROOT}/design.json"
