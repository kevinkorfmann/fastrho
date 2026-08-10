#!/usr/bin/env bash
#SBATCH --job-name=arabis-prep
#SBATCH --partition=genoa-std-mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
"${ARABIS_PIXI}" install --manifest-path "${ARABIS_REPO}/research/arabis/pixi.toml"
cp "${ARABIS_REPO}/scripts/arabis_wgs_stage.sh" "${ARABIS_NODE_TMP}/arabis_wgs_stage.sh"
arabis_bio bash "${ARABIS_NODE_TMP}/arabis_wgs_stage.sh" prepare
