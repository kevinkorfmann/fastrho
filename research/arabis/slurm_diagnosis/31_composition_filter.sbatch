#!/usr/bin/env bash
#SBATCH --job-name=arabis-composition-vcf
#SBATCH --partition=genoa-std-mem
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
root="${ARABIS_ROOT}/diagnosis/composition"
mkdir -p "${root}/vcf"
printf 'panel\tspecies\tdesign\tomitted\taccessions\n' > "${root}/panels.tsv"
common=(10 25 29 30 337 358)
pairs=('163 233' '293 309' '393 435')
for i in 0 1 2; do
  panel="nemorensis_rhine6_pair${i}"
  read -r a b <<< "${pairs[$i]}"
  printf '%s\n' "${common[@]}" "${a}" "${b}" > "${root}/${panel}.samples"
  printf '%s\tnemorensis\trhine6_plus_disjoint_allopatric_pair\t\t%s\n' \
    "${panel}" "$(paste -sd, "${root}/${panel}.samples")" >> "${root}/panels.tsv"
  out="${root}/vcf/${panel}.selfer.vcf.gz"
  arabis_bio bcftools view -S "${root}/${panel}.samples" -Ou "${ARABIS_WORK}/vcf/all.genome.vcf.gz" |
    arabis_bio bcftools +setGT -Ou -- -t q -n . -i 'FMT/DP<5 || FMT/DP>60' |
    arabis_bio bcftools view -i 'F_MISSING=0 && COUNT(GT="het")=0 && MAF>=0.05' -Oz -o "${out}.partial"
  mv "${out}.partial" "${out}"
  arabis_bio bcftools index -f "${out}"
done
