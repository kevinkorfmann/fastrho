#!/usr/bin/env bash
#SBATCH --job-name=arabis-resample-vcf
#SBATCH --partition=genoa-std-mem
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%A_%a.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
root="${ARABIS_ROOT}/diagnosis/resamples"
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${root}/panels.tsv")
IFS=$'\t' read -r panel _ _ _ _ <<< "${line}"
[[ -n "${panel}" ]]
out="${root}/vcf/${panel}.selfer.vcf.gz"
mkdir -p "${root}/vcf"
arabis_bio bcftools view -S "${root}/${panel}.samples" -Ou "${ARABIS_WORK}/vcf/all.genome.vcf.gz" |
  arabis_bio bcftools +setGT -Ou -- -t q -n . -i 'FMT/DP<5 || FMT/DP>60' |
  arabis_bio bcftools view -i 'F_MISSING=0 && COUNT(GT="het")=0 && MAF>=0.05' -Oz -o "${out}.partial"
mv "${out}.partial" "${out}"
arabis_bio bcftools index -f "${out}"
printf '%s\t%s\n' "${panel}" "$(arabis_bio bcftools index -n "${out}")"
