#!/usr/bin/env bash
#SBATCH --job-name=arabis-filter
#SBATCH --partition=genoa-std-mem
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
cp "${ARABIS_REPO}/scripts/arabis_wgs_stage.sh" "${ARABIS_NODE_TMP}/arabis_wgs_stage.sh"
arabis_bio bash "${ARABIS_NODE_TMP}/arabis_wgs_stage.sh" finalize
