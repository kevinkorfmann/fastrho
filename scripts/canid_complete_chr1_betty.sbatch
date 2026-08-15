#!/bin/bash
#SBATCH --job-name=canid-complete
#SBATCH --partition=genoa-std-mem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=01:00:00
#SBATCH --output=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/complete-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/complete-%j.err

set -euo pipefail
source /etc/profile.d/Z98-lmod.sh 2>/dev/null || true
module load python/3.11.11 bcftools/1.21

analysis_project=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33
analysis_code=$analysis_project/repo
analysis_python=/vast/projects/smathi/cohort/korfmann/fastrho-arabis/python/bin/python
analysis_vcf=https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz

"$analysis_python" "$analysis_code/scripts/canid_multi_panel_from_vcf.py" "$analysis_vcf" \
  --panel wolf33_chr1_complete "$analysis_project/inputs/wolf33_ids.txt" \
    "$analysis_project/hap/wolf33_chr1_complete.npz" \
  --panel dog42_chr1_complete "$analysis_project/inputs/dog_village42_ids.txt" \
    "$analysis_project/hap/dog42_chr1_complete.npz" \
  --region chr1:1-123000000 --missing-policy complete
echo "finished $(date -Is)"
