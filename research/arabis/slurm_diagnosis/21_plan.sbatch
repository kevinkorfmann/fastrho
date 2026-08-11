#!/usr/bin/env bash
#SBATCH --job-name=arabis-resample-plan
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
out="${ARABIS_ROOT}/diagnosis/resamples"
mkdir -p "${out}"
"${ARABIS_ENV}/bin/python" "${ARABIS_REPO}/scripts/arabis_subset_plan.py" \
  --sample-sheet "${ARABIS_REPO}/research/arabis/sample_assignments.tsv" \
  --output-dir "${out}"
