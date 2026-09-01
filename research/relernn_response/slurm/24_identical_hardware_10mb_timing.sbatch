#!/usr/bin/env bash
#SBATCH --job-name=rho-time-10mb
#SBATCH --partition=dgx-b200
#SBATCH --exclude=dgx015
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#SBATCH --time=24:00:00
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-timing-identical-10mb-20260901/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-timing-identical-10mb-20260901/logs/%x-%j.err
set -euo pipefail

export TIMING_ROOT=/vast/projects/smathi/cohort/${USER}/fastrho-timing-identical-10mb-20260901
export TIMING_SOURCE=/vast/projects/smathi/cohort/${USER}/fastrho-relernn-intended-20260831
export RESPONSE_ROOT=${TIMING_ROOT}
export RESPONSE_SOURCE_ROOT=/vast/projects/smathi/cohort/${USER}/fastrho-demography-v1
export RESPONSE_REPO=${SLURM_SUBMIT_DIR}
export PATH=/vast/parcc/spack/sw/apps/linux-sapphirerapids/apptainer-1.4.1-qpr4lterya7ontg7pkqrx7b3jkab3lcw/bin:${PATH}

if [[ -e "${TIMING_ROOT}/.started" ]]; then
  echo "Refusing to overwrite an existing timing run: ${TIMING_ROOT}" >&2
  exit 2
fi
touch "${TIMING_ROOT}/.started"
mkdir -p "${TIMING_ROOT}/arm" "${TIMING_ROOT}/hardware" "${TIMING_ROOT}/results" "${TIMING_ROOT}/cache"

source_arm=${TIMING_SOURCE}/arms/bottleneck
timing_arm=${TIMING_ROOT}/arm
cp -al "${source_arm}/config.json" "${source_arm}/genome.bed" "${timing_arm}/"
for suffix in vcf trees npz; do
  cp -al "${source_arm}"/region_*."${suffix}" "${timing_arm}/"
done

hostname > "${TIMING_ROOT}/hardware/hostname.txt"
printf '%s\n' "${SLURM_CPUS_PER_TASK}" > "${TIMING_ROOT}/hardware/allocated_cpus.txt"
printf '%s\n' "${SLURM_JOB_ID}" > "${TIMING_ROOT}/hardware/slurm_job_id.txt"
lscpu > "${TIMING_ROOT}/hardware/lscpu.txt"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader > "${TIMING_ROOT}/hardware/gpu.csv"
env | sort > "${TIMING_ROOT}/hardware/environment.txt"

# fastrho: checkpoint loading and prediction on the shared B200 allocation.
source /etc/profile.d/Z98-lmod.sh 2>/dev/null || true
module load python/3.11.11 cuda/12.8.1
fastrho_python=/vast/projects/smathi/cohort/${USER}/fastrho-arabis/python/bin/python
export PYTHONPATH=${RESPONSE_REPO}${PYTHONPATH:+:${PYTHONPATH}}
export TRITON_CACHE_DIR=${TIMING_ROOT}/cache/triton
export NUMBA_CACHE_DIR=${TIMING_ROOT}/cache/numba
mkdir -p "${TRITON_CACHE_DIR}" "${NUMBA_CACHE_DIR}"
"${fastrho_python}" "${RESPONSE_REPO}/scripts/bench.py" fastrho \
  --config "${timing_arm}" \
  --checkpoint "${TIMING_SOURCE}/reference/fastrho.ckpt" \
  --stats "${TIMING_SOURCE}/reference/feat_stats.npz" \
  --device cuda:0

# pyrho: matched-demography lookup construction and inference on the same node.
pyrho_python=${RESPONSE_SOURCE_ROOT}/envs/pyrho-py312/bin/python
export PYRHO_TABLE_THREADS=${SLURM_CPUS_PER_TASK}
export PYRHO_OPTIMIZE_THREADS=8
"${pyrho_python}" "${RESPONSE_REPO}/scripts/run_pyrho_config.py" "${timing_arm}"

# ReLERNN: matched-demography simulation, 100-epoch training, and prediction.
source "${RESPONSE_REPO}/research/relernn_response/slurm/common.sh"
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1
relernn_run python "${RESPONSE_REPO}/scripts/run_relernn_paired.py" simulate \
  --config-dir "${timing_arm}" \
  --demography-history "${RESPONSE_REPO}/research/demography_matched/demographies/bottleneck_smcpp.csv" \
  --n-cpu "${SLURM_CPUS_PER_TASK}" \
  --n-train 100000 --n-validation 10000 --n-test 10000 \
  --max-sites 1750 --upper-rho-theta-ratio 3 --seed 1
relernn_run python "${RESPONSE_REPO}/scripts/run_relernn_paired.py" train-predict \
  --config-dir "${timing_arm}" --train-cpu 8 --epochs 100 --seed 1

"${pyrho_python}" "${RESPONSE_REPO}/scripts/collect_identical_hardware_timing.py" \
  --root "${TIMING_ROOT}" \
  --source-arm "${source_arm}" \
  --output "${TIMING_ROOT}/results/identical_hardware_10mb_timing.json"
touch "${TIMING_ROOT}/.complete"
