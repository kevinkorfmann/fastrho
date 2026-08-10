#!/usr/bin/env bash
#SBATCH --job-name=demo-pyrho
#SBATCH --partition=genoa-std-mem
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"

arms=(bottleneck_pyrho_constant bottleneck_pyrho_matched expansion_pyrho_constant expansion_pyrho_matched)
arm=${arms[${SLURM_ARRAY_TASK_ID}]}
config="${DEMO_ROOT}/arms/${arm}"
"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/run_pyrho_config.py" "${config}"
"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/bench.py" ingest --config "${config}" --kind pyrho
"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/bench.py" score \
  --config "${config}" --methods pyrho --results "${DEMO_ROOT}/results"
