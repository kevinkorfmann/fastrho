#!/usr/bin/env bash
#SBATCH --job-name=arabis-mac2-ensemble
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
root="${ARABIS_ROOT}/diagnosis/mac2"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_ensemble_maps.py" \
  --input-root "${root}/maps/nemorensis_mac2" \
  --manifest "${ARABIS_SMALLN_ROOT}/frozen_ensemble.json" \
  --output-dir "${root}/maps/nemorensis_mac2/ensemble" --members 5 --expected-files 8
