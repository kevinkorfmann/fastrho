#!/bin/bash
#SBATCH --job-name=canid-full33
#SBATCH --partition=b200-mig45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:45gb:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/run-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33/logs/run-%j.err

set -euo pipefail

source /etc/profile.d/Z98-lmod.sh 2>/dev/null || true
module load python/3.11.11 cuda/12.8.1 bcftools/1.21

PROJECT=/vast/projects/smathi/cohort/korfmann/fastrho-canid-full33
CODE=$PROJECT/repo
PY=/vast/projects/smathi/cohort/korfmann/fastrho-arabis/python/bin/python
VCF=https://research.nhgri.nih.gov/dog_genome/downloads/datasets/WGS/722g.990.SNP.INDEL.chrAll.vcf.gz
IDS=$PROJECT/inputs/wolf33_ids.txt
export FASTRHO_REALDATA_ROOT=$PROJECT
export STDPOPSIM_CACHE=$PROJECT/cache/stdpopsim
export PYTHONPATH=$CODE${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=$PROJECT/cache/triton
export NUMBA_CACHE_DIR=$PROJECT/cache/numba
mkdir -p "$PROJECT"/{hap,maps,logs} "$TRITON_CACHE_DIR" "$NUMBA_CACHE_DIR"

echo "started $(date -Is) on $(hostname)"
nvidia-smi -L
"$PY" -c 'import numpy, scipy, stdpopsim, torch, mamba_ssm; print(numpy.__version__, scipy.__version__, stdpopsim.__version__, torch.__version__)'

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

"$PY" "$CODE/scripts/wolf_subset_infer.py" \
  wolf33_full_reference "$PROJECT/inputs/dogbn.ckpt" \
  "$PROJECT/inputs/dogbn-feat-stats.npz" \
  "$PROJECT/maps/wolf33_full_reference_dogbn.npz" cuda:0

"$PY" "$CODE/scripts/canid_empirical_bootstrap.py" \
  --wolf "$PROJECT/maps/wolf33_full_reference_dogbn.npz" \
  --dog all_dogs "$PROJECT/maps/dog67_full_reference_dogbn.npz" \
  --dog village_dogs "$PROJECT/maps/dog_village42_full_reference_dogbn.npz" \
  --output "$PROJECT/maps/wolf33_full_reference_bootstrap.json"

echo "finished $(date -Is)"
