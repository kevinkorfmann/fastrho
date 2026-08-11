#!/usr/bin/env bash
#SBATCH --job-name=arabis-align
#SBATCH --partition=genoa-std-mem
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=12
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
cp "${ARABIS_REPO}/scripts/arabis_wgs_stage.sh" "${ARABIS_NODE_TMP}/arabis_wgs_stage.sh"
arabis_bio bash "${ARABIS_NODE_TMP}/arabis_wgs_stage.sh" align
