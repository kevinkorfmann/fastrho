#!/usr/bin/env bash
#SBATCH --job-name=demo-fastrho-reference
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"
cp "${DEMO_REPO}/research/demography_matched/fastrho_reference.json" \
  "${DEMO_ROOT}/fastrho_reference.json"
"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/score_fastrho_reference.py" \
  --root "${DEMO_ROOT}" --manifest "${DEMO_ROOT}/fastrho_reference.json" \
  --output "${DEMO_ROOT}/results/fastrho_reference.json"
