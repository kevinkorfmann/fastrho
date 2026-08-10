#!/usr/bin/env bash
#SBATCH --job-name=struct-prereg
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
cp "${ARABIS_REPO}/research/arabis/structured_selfing_preregistration.json" \
  "${ARABIS_STRUCTURED_ROOT}/preregistration.json"
sha256sum \
  "${ARABIS_STRUCTURED_ROOT}/preregistration.json" \
  "${ARABIS_REPO}/scripts/arabis_structured_selfing_gen.py" \
  "${ARABIS_REPO}/scripts/arabis_simulation_gate.py" \
  > "${ARABIS_STRUCTURED_ROOT}/preregistered_inputs.sha256"
date --utc --iso-8601=seconds > "${ARABIS_STRUCTURED_ROOT}/preregistered_utc.txt"
