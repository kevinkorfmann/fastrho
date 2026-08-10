#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; echo "arabis_stage=${STAGE:-unset} failed_line=${LINENO} status=${status}" >&2; exit "${status}"' ERR

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Use the Betty Slurm workflow in research/arabis/slurm/submit.sh" >&2
  exit 2
}

STAGE=${1:?stage required: prepare, download, align, call, or finalize}
WORK=${ARABIS_WORK:?ARABIS_WORK must be exported}
REPO=${ARABIS_REPO:?ARABIS_REPO must be exported}
REF=${WORK}/reference/arabis_nemorensis_GCA_976985625.1.fa
SAMPLE_SHEET=${REPO}/research/arabis/sample_assignments.tsv
THREADS=${SLURM_CPUS_PER_TASK:-4}

mkdir -p "${WORK}"/{reference,raw,bam,vcf,logs,maps}

fetch_fastq() {
  local url=$1 dest=$2 expected=$3 partial="${2}.partial"
  if [[ -s "${dest}" ]] && [[ $(stat -c '%s' "${dest}") == "${expected}" ]] && gzip -t "${dest}"; then
    return 0
  fi
  if [[ -e "${dest}" && ! -e "${partial}" ]]; then
    mv "${dest}" "${partial}"
  fi
  if [[ -e "${partial}" ]] && [[ $(stat -c '%s' "${partial}") == "${expected}" ]]; then
    if gzip -t "${partial}"; then
      mv "${partial}" "${dest}"
      return 0
    fi
    mv "${partial}" "${partial}.corrupt.$(date +%s)"
  fi
  curl -L --fail --retry 12 --retry-all-errors --connect-timeout 60 \
    --speed-limit 1024 --speed-time 120 --silent --show-error -C - \
    -o "${partial}" "https://${url}"
  [[ $(stat -c '%s' "${partial}") == "${expected}" ]]
  gzip -t "${partial}"
  mv "${partial}" "${dest}"
}

case "${STAGE}" in
  prepare)
    MANIFEST="${WORK}/ena_PRJEB39992.tsv"
    REFERENCE_MANIFEST="${WORK}/ena_PRJEB33482_reference.tsv"
    if [[ ! -s "${MANIFEST}" ]]; then
      curl -L --fail --retry 5 --silent --show-error \
        'https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=study_accession%3D%22PRJEB39992%22&fields=run_accession,sample_accession,scientific_name,sample_alias,library_strategy,library_layout,read_count,base_count,fastq_ftp,fastq_bytes&format=tsv&limit=0' \
        -o "${MANIFEST}.partial"
      mv "${MANIFEST}.partial" "${MANIFEST}"
    fi
    if [[ ! -s "${REFERENCE_MANIFEST}" ]]; then
      curl -L --fail --retry 5 --silent --show-error \
        'https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=study_accession%3D%22PRJEB33482%22&fields=run_accession,sample_accession,scientific_name,sample_alias,library_strategy,library_layout,read_count,base_count,fastq_ftp,fastq_bytes&format=tsv&limit=0' \
        -o "${REFERENCE_MANIFEST}.partial"
      mv "${REFERENCE_MANIFEST}.partial" "${REFERENCE_MANIFEST}"
    fi

    if [[ ! -s "${REF}" ]]; then
      : > "${REF}.partial"
      accessions=(OZ335658 OZ335659 OZ335660 OZ335661 OZ335662 OZ335663 OZ335664 OZ335665)
      for i in "${!accessions[@]}"; do
        curl -L --fail --retry 5 --silent --show-error \
          "https://www.ebi.ac.uk/ena/browser/api/fasta/${accessions[$i]}?download=true" |
          awk -v h="chr$((i + 1))" '/^>/{print ">" h; next}{print}' >> "${REF}.partial"
      done
      mv "${REF}.partial" "${REF}"
    fi
    [[ -s "${REF}.fai" ]] || samtools faidx "${REF}"
    [[ -s "${REF}.bwt.2bit.64" ]] || bwa-mem2 index "${REF}"

    awk -F '\t' -v OFS='\t' -v reference_manifest="${REFERENCE_MANIFEST}" '
      NR==FNR {
        if (FNR > 1) { species[$1]=$2; assigned[$1]=$3; population[$1]=$4 }
        next
      }
      FNR==1 { next }
      {
        accession=$4; sub(/_[12]$/, "", accession)
        selected_reference=(FILENAME != reference_manifest || $1 == "ERR3454147")
        if ($5=="WGS" && $6=="PAIRED" && accession in species && selected_reference) {
          print $1,$2,species[accession],$4,$5,$6,$7,$8,$9,$10,accession,assigned[accession],population[accession]
        }
      }
    ' "${SAMPLE_SHEET}" "${MANIFEST}" "${REFERENCE_MANIFEST}" > "${WORK}/wgs.tsv"
    [[ $(wc -l < "${WORK}/wgs.tsv") -eq 37 ]]
    [[ $(cut -f11 "${WORK}/wgs.tsv" | sort -u | wc -l) -eq 37 ]]
    [[ $(awk -F '\t' '$3=="nemorensis" {a[$11]=1} END {print length(a)}' "${WORK}/wgs.tsv") -eq 12 ]]
    [[ $(awk -F '\t' '$3=="sagittata" {a[$11]=1} END {print length(a)}' "${WORK}/wgs.tsv") -eq 25 ]]

    CROSS_MAP=${WORK}/map6j_742_2082_gen.csv
    if ! echo 'b980746fe7ffc1b5bbb7691d131bdd8329f8e427995abd2786185383c5273bed  '"${CROSS_MAP}" | sha256sum -c - >/dev/null 2>&1; then
      curl -L --fail --retry 5 --silent --show-error \
        'https://raw.githubusercontent.com/nedarahnama/Contemporary_hybridization/10c9092ce9e08c16b0a958d4062932262a8c6bc4/04_genetic_map/map6j_742_2082_gen.csv' \
        -o "${CROSS_MAP}.partial"
      echo 'b980746fe7ffc1b5bbb7691d131bdd8329f8e427995abd2786185383c5273bed  '"${CROSS_MAP}.partial" | sha256sum -c -
      mv "${CROSS_MAP}.partial" "${CROSS_MAP}"
    fi
    ;;
  download)
    : "${SLURM_ARRAY_TASK_ID:?download must be submitted as an array}"
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${WORK}/wgs.tsv")
    [[ -n "${line}" ]]
    IFS=$'\t' read -r run _ _ _ _ _ _ _ ftp bytes _ _ _ <<< "${line}"
    IFS=';' read -r url1 url2 <<< "${ftp}"
    IFS=';' read -r size1 size2 <<< "${bytes}"
    fetch_fastq "${url1}" "${WORK}/raw/${run}_1.fastq.gz" "${size1}"
    fetch_fastq "${url2}" "${WORK}/raw/${run}_2.fastq.gz" "${size2}"
    ;;
  align)
    : "${SLURM_ARRAY_TASK_ID:?align must be submitted as an array}"
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${WORK}/wgs.tsv")
    [[ -n "${line}" ]]
    IFS=$'\t' read -r run _ _ _ _ _ _ _ _ _ accession _ _ <<< "${line}"
    out="${WORK}/bam/${run}.bam"
    if [[ -s "${out}.bai" ]] && samtools quickcheck "${out}"; then exit 0; fi
    gzip -t "${WORK}/raw/${run}_1.fastq.gz" "${WORK}/raw/${run}_2.fastq.gz"
    bwa-mem2 mem -t "${THREADS}" -R "@RG\tID:${run}\tSM:${accession}\tPL:ILLUMINA" \
      "${REF}" "${WORK}/raw/${run}_1.fastq.gz" "${WORK}/raw/${run}_2.fastq.gz" |
      samtools sort -@ 4 -m 3G -o "${out}.partial" -
    samtools quickcheck "${out}.partial"
    mv "${out}.partial" "${out}"
    samtools index -@ 4 "${out}"
    ;;
  call)
    : "${SLURM_ARRAY_TASK_ID:?call must be submitted as an array}"
    chunk_bp=5000000
    regions="${ARABIS_NODE_TMP:?ARABIS_NODE_TMP must be exported}/call_regions.tsv"
    awk -v chunk="${chunk_bp}" -v OFS='\t' '
      {
        part=0
        for (start=1; start <= $2; start += chunk) {
          end=start + chunk - 1
          if (end > $2) end=$2
          print $1,start,end,part
          part++
        }
      }
    ' "${REF}.fai" > "${regions}"
    line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${regions}")
    [[ -n "${line}" ]]
    IFS=$'\t' read -r chrom start end part <<< "${line}"
    bamlist="${ARABIS_NODE_TMP:?ARABIS_NODE_TMP must be exported}/bamlist.${chrom}.txt"
    find "${WORK}/bam" -name '*.bam' -type f | sort > "${bamlist}"
    [[ $(wc -l < "${bamlist}") -eq 37 ]]
    printf -v part_tag '%03d' "${part}"
    out="${WORK}/vcf/all.${chrom}.part${part_tag}.vcf.gz"
    bcftools mpileup -Ou -f "${REF}" -r "${chrom}:${start}-${end}" -q 30 -Q 20 -d 500 \
      -a FORMAT/DP,FORMAT/AD -b "${bamlist}" --threads "${THREADS}" |
      bcftools call -mv -Ou --threads "${THREADS}" |
      bcftools view -m2 -M2 -v snps -Oz --threads "${THREADS}" -o "${out}.partial"
    mv "${out}.partial" "${out}"
    bcftools index -f "${out}"
    ;;
  finalize)
    for chrom in {1..8}; do
      mapfile -t parts < <(find "${WORK}/vcf" -name "all.chr${chrom}.part*.vcf.gz" -type f | sort)
      expected=$(awk -v c="chr${chrom}" '$1==c {print int(($2 + 4999999) / 5000000)}' "${REF}.fai")
      [[ ${#parts[@]} -eq ${expected} ]]
      for part in "${parts[@]}"; do [[ -s "${part}.csi" ]]; done
      bcftools concat -a -Oz --threads "${THREADS}" \
        -o "${WORK}/vcf/all.chr${chrom}.vcf.gz.partial" "${parts[@]}"
      mv "${WORK}/vcf/all.chr${chrom}.vcf.gz.partial" "${WORK}/vcf/all.chr${chrom}.vcf.gz"
      bcftools index -f "${WORK}/vcf/all.chr${chrom}.vcf.gz"
    done
    bcftools concat -a -Oz --threads "${THREADS}" -o "${WORK}/vcf/all.genome.vcf.gz.partial" \
      "${WORK}"/vcf/all.chr{1,2,3,4,5,6,7,8}.vcf.gz
    mv "${WORK}/vcf/all.genome.vcf.gz.partial" "${WORK}/vcf/all.genome.vcf.gz"
    bcftools index -f "${WORK}/vcf/all.genome.vcf.gz"
    [[ $(bcftools query -l "${WORK}/vcf/all.genome.vcf.gz" | wc -l) -eq 37 ]]
    for species in nemorensis sagittata; do
      awk -F '\t' -v s="${species}" '$3==s {print $11}' "${WORK}/wgs.tsv" | sort -u > "${WORK}/${species}.samples"
      parent=10; [[ "${species}" == sagittata ]] && parent=69
      grep -vx "${parent}" "${WORK}/${species}.samples" > "${WORK}/${species}.no_cross_parent.samples"
      for suffix in '' '.no_cross_parent'; do
        samples="${WORK}/${species}${suffix}.samples"
        out="${WORK}/vcf/${species}${suffix}.selfer.vcf.gz"
        bcftools view -S "${samples}" -Ou "${WORK}/vcf/all.genome.vcf.gz" |
          bcftools +setGT -Ou -- -t q -n . -i 'FMT/DP<5 || FMT/DP>60' |
          bcftools view -i 'F_MISSING=0 && COUNT(GT="het")=0 && MAF>=0.05' -Oz -o "${out}.partial"
        mv "${out}.partial" "${out}"
        bcftools index -f "${out}"
        [[ $(bcftools query -l "${out}" | wc -l) -eq $(wc -l < "${samples}") ]]
      done
    done
    {
      printf 'bwa-mem2\t'; bwa-mem2 version 2>&1 | tail -n 1
      printf 'samtools\t'; samtools --version | sed -n '1p'
      printf 'bcftools\t'; bcftools --version | sed -n '1p'
    } > "${WORK}/software_versions.tsv"
    bcftools stats "${WORK}/vcf/nemorensis.selfer.vcf.gz" > "${WORK}/vcf/nemorensis.stats.txt"
    bcftools stats "${WORK}/vcf/sagittata.selfer.vcf.gz" > "${WORK}/vcf/sagittata.stats.txt"
    ;;
  *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
