#!/usr/bin/env bash
#SBATCH --job-name=demo-archive-pilot
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"

archive="${DEMO_ROOT}/technical_pilot_invalid_gt_20260721"
[[ ! -e "${archive}" ]] || {
  echo "Refusing to overwrite technical pilot archive: ${archive}" >&2
  exit 2
}
mkdir -p "${archive}"

for name in arms validated_input results runtime smoke input_manifest.json design.json submission.tsv; do
  if [[ -e "${DEMO_ROOT}/${name}" ]]; then
    mv "${DEMO_ROOT}/${name}" "${archive}/${name}"
  fi
done
mkdir -p "${DEMO_ROOT}/results" "${DEMO_ROOT}/runtime"

printf '%s\n' \
  'Archived before outcome-blind rerun because 20 raw VCF rows contained GT allele 2 despite a single ALT allele.' \
  'The replacement preparation drops every row containing a GT allele outside {0,1} before creating any method arm.' \
  > "${archive}/TECHNICAL_REASON.txt"
