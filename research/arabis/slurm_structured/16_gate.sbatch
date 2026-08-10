#!/usr/bin/env bash
#SBATCH --job-name=struct-gate
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_simulation_gate.py" \
  --manifest "${ARABIS_STRUCTURED_ROOT}/frozen_ensemble.json" \
  --audit-dir "${ARABIS_STRUCTURED_ROOT}/simulation_audit" \
  --preregistration "${ARABIS_STRUCTURED_ROOT}/preregistration.json" \
  --output "${ARABIS_STRUCTURED_ROOT}/simulation_gate.json"
