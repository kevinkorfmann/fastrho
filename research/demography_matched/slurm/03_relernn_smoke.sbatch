#!/usr/bin/env bash
#SBATCH --job-name=demo-smoke
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=1

relernn_run python - <<'PY'
import tensorflow as tf
devices = tf.config.list_physical_devices("GPU")
assert devices, "TensorFlow does not see the Betty B200 allocation"
print(tf.__version__)
print(devices)
PY

smoke="${DEMO_ROOT}/smoke"
if [[ ! -e "${smoke}/DONE" ]]; then
  rm -rf "${smoke}"
  mkdir -p "${smoke}"
  cp "${DEMO_RELERNNSRC}/examples/example.vcf" "${smoke}/smoke.vcf"
  cp "${DEMO_RELERNNSRC}/examples/genome.bed" "${smoke}/genome.bed"
  relernn_run ReLERNN_SIMULATE --vcf "${smoke}/smoke.vcf" --genome "${smoke}/genome.bed" \
    --projectDir "${smoke}/project" --assumedMu 1e-8 --assumedGenTime 1 \
    --upperRhoThetaRatio 10 --nCPU "${SLURM_CPUS_PER_TASK}" --seed 1 --phased --maxSites 256 \
    --nTrain 6400 --nVali 64 --nTest 64
  relernn_run ReLERNN_TRAIN --projectDir "${smoke}/project" --nEpochs 1 --nCPU 4 --seed 1 --gpuID 0
  relernn_run ReLERNN_PREDICT --vcf "${smoke}/smoke.vcf" --projectDir "${smoke}/project" \
    --phased --seed 1 --gpuID 0
  compgen -G "${smoke}/project/*PREDICT*txt" >/dev/null
  touch "${smoke}/DONE"
fi
