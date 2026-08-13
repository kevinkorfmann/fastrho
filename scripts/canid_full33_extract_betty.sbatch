#!/bin/bash
#SBATCH --job-name=canid33-extract
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=02:00:00
#SBATCH --output=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/extract-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/extract-%j.err

set -euo pipefail
source /etc/profile.d/Z98-lmod.sh 2>/dev/null || true
module load python/3.11.11 bcftools/1.21

PROJECT=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33
CODE=$PROJECT/repo
PY=/vast/projects/smathi/cohort/korfmann/fastrho-arabis/python/bin/python
VCF_URL=https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz
VCF_INDEX=$PROJECT/inputs/722g.990.SNP.INDEL.chrAll.vcf.gz.tbi
VCF=${VCF_URL}##idx##${VCF_INDEX}
IDS=$PROJECT/inputs/wolf33_ids.txt
mkdir -p "$PROJECT"/{hap,logs}

regions=(
  chr1:1-15000000
  chr1:15000001-30000000
  chr1:30000001-45000000
  chr1:45000001-60000000
  chr1:60000001-75000000
  chr1:75000001-90000000
  chr1:90000001-105000000
  chr1:105000001-123000000
)

echo "started $(date -Is) on $(hostname)"
pids=()
chunks=()
for offset in 0 1 2 3 4 5 6 7; do
  index=$(printf '%02d' "$((offset + 1))")
  output=$PROJECT/hap/wolf33_ref_chunk${index}.npz
  log=$PROJECT/logs/extract-chunk${index}.log
  chunks+=("$output")
  if test -s "$output"; then
    continue
  fi
  "$PY" "$CODE/scripts/canid_multi_panel_from_vcf.py" "$VCF" \
    --panel wolf33 "$IDS" "$output" \
    --region "${regions[$offset]}" --missing-policy reference >"$log" 2>&1 &
  pids+=("$!")
  if test "${#pids[@]}" -eq 4; then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" "$CODE/scripts/canid_merge_npz.py" \
  "$PROJECT/hap/wolf33_full_reference.npz" "${chunks[@]}" \
  --key wolf33_full_reference
echo "finished $(date -Is)"
