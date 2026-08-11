#!/usr/bin/env bash
#SBATCH --job-name=arabis-nem-jackknife-eval
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm_smalln/common.sh"
root="${ARABIS_ROOT}/diagnosis/resamples"
head -n 13 "${root}/panels.tsv" > "${root}/nemorensis_jackknife_panels.tsv"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_subset_evaluate.py" \
  --manifest "${root}/nemorensis_jackknife_panels.tsv" --maps-root "${root}/maps" \
  --full-maps "${ARABIS_SMALLN_ROOT}/arabis_maps/ensemble" \
  --cross-map "${ARABIS_WORK}/map6j_742_2082_gen.csv" \
  --output "${ARABIS_ROOT}/diagnosis/nemorensis_jackknife_results.json"
