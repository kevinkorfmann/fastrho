#!/usr/bin/env bash
#SBATCH --job-name=arabis-panel-stats
#SBATCH --partition=genoa-std-mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_panel_diagnostics.py" \
  --workdir "${ARABIS_WORK}" \
  --sample-sheet "${ARABIS_REPO}/research/arabis/sample_assignments.tsv" \
  --output "${ARABIS_ROOT}/diagnosis/panel_stats.json"
