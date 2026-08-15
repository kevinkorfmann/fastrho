#!/bin/bash
#SBATCH --job-name=canid-multichrom
#SBATCH --partition=b200-mig45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=2-5%4
#SBATCH --output=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/multichrom-%A_%a.out
#SBATCH --error=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/multichrom-%A_%a.err

set -euo pipefail
source /etc/profile.d/Z98-lmod.sh 2>/dev/null || true
module load python/3.11.11 cuda/12.8.1 bcftools/1.21

analysis_project=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33
analysis_code=$analysis_project/repo
analysis_python=/vast/projects/smathi/cohort/korfmann/fastrho-arabis/python/bin/python
analysis_vcf=https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz
analysis_chrom=$SLURM_ARRAY_TASK_ID
analysis_wolf_key=wolf33_chr${analysis_chrom}
analysis_dog_key=dog42_chr${analysis_chrom}
analysis_wolf_hap=$analysis_project/hap/${analysis_wolf_key}.npz
analysis_dog_hap=$analysis_project/hap/${analysis_dog_key}.npz
analysis_wolf_map=$analysis_project/maps/${analysis_wolf_key}_dogbn.npz
analysis_dog_map=$analysis_project/maps/${analysis_dog_key}_dogbn.npz
export FASTRHO_REALDATA_ROOT=$analysis_project
export STDPOPSIM_CACHE=$analysis_project/cache/stdpopsim
export PYTHONPATH=$analysis_code${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=$analysis_project/cache/triton
export NUMBA_CACHE_DIR=$analysis_project/cache/numba
mkdir -p "$analysis_project"/{hap,maps,logs} "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR"

echo "started $(date -Is) on $(hostname), chromosome $analysis_chrom"
if ! test -s "$analysis_wolf_hap" || ! test -s "$analysis_dog_hap"; then
  "$analysis_python" "$analysis_code/scripts/canid_multi_panel_from_vcf.py" "$analysis_vcf" \
    --panel "$analysis_wolf_key" "$analysis_project/inputs/wolf33_ids.txt" "$analysis_wolf_hap" \
    --panel "$analysis_dog_key" "$analysis_project/inputs/dog_village42_ids.txt" "$analysis_dog_hap" \
    --region "chr${analysis_chrom}:1-200000000" --missing-policy reference
fi

if ! test -s "$analysis_wolf_map"; then
  "$analysis_python" "$analysis_code/scripts/wolf_subset_infer.py" \
    "$analysis_wolf_key" "$analysis_project/inputs/dogbn.ckpt" \
    "$analysis_project/inputs/dogbn-feat-stats.npz" "$analysis_wolf_map" cuda:0
fi
if ! test -s "$analysis_dog_map"; then
  "$analysis_python" "$analysis_code/scripts/wolf_subset_infer.py" \
    "$analysis_dog_key" "$analysis_project/inputs/dogbn.ckpt" \
    "$analysis_project/inputs/dogbn-feat-stats.npz" "$analysis_dog_map" cuda:0
fi
echo "finished $(date -Is)"
