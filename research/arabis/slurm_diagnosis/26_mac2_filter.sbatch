#!/usr/bin/env bash
#SBATCH --job-name=arabis-mac2-vcf
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-arabis/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/arabis/slurm/common.sh"
root="${ARABIS_ROOT}/diagnosis/mac2"
mkdir -p "${root}/vcf"
out="${root}/vcf/nemorensis_mac2.selfer.vcf.gz"
# Diploid homozygotes contribute two VCF alleles, so MAC>=4 means at least
# two predominantly selfing accessions carry the minor state.
arabis_bio bcftools view -i 'MAC>=4' -Oz -o "${out}.partial" \
  "${ARABIS_WORK}/vcf/nemorensis.selfer.vcf.gz"
mv "${out}.partial" "${out}"
arabis_bio bcftools index -f "${out}"
printf 'panel\tspecies\tdesign\tomitted\taccessions\n' > "${root}/panels.tsv"
printf 'nemorensis_mac2\tnemorensis\tminimum_minor_allele_count_2\t\tall_12\n' >> "${root}/panels.tsv"
printf 'retained_sites\t%s\n' "$(arabis_bio bcftools index -n "${out}")"
