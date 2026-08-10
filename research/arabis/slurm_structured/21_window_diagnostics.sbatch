#!/usr/bin/env bash
#SBATCH --job-name=struct-window-audit
#SBATCH --partition=genoa-std-mem
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_structured/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_window_diagnostics.py" \
  --campaign-root "${ARABIS_STRUCTURED_ROOT}" \
  --workdir "${ARABIS_WORK}" \
  --windows-npz "${ARABIS_STRUCTURED_ROOT}/arabis_cross_windows.npz" \
  --output "${ARABIS_STRUCTURED_ROOT}/arabis_window_diagnostics.json" \
  --members 7 --context-len 1024 --overlap 256
