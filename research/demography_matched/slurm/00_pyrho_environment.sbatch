#!/usr/bin/env bash
#SBATCH --job-name=demo-pyrho-env
#SBATCH --partition=genoa-std-mem
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"

if [[ ! -x "${DEMO_PYRHO_ENV}/bin/python" ]]; then
  mkdir -p "$(dirname "${DEMO_PYRHO_ENV}")"
  "${DEMO_UV}" venv --python 3.12 "${DEMO_PYRHO_ENV}"
fi
"${DEMO_UV}" pip install --python "${DEMO_PYRHO_ENV}/bin/python" \
  "numpy==1.26.4" "scipy==1.12.0" "msprime==1.3.4" \
  "scikit-learn>=1.4,<2" "numba==0.60.0" "llvmlite==0.43.0" \
  "pandas==2.1.4" "tables==3.11.1" "cyvcf2==0.31.4" \
  "ldpop @ git+https://github.com/popgenmethods/ldpop@cceed0150194ea8bf3415aa9bf3ee002e6b6e480" \
  "pyrho @ git+https://github.com/popgenmethods/pyrho@fe7d1cfe1e5b10bee1f1d0828dbf1dc5391d9888"

"${DEMO_PYRHO_ENV}/bin/python" "${DEMO_REPO}/scripts/capture_demography_runtime.py" \
  --mode pyrho --output "${DEMO_ROOT}/runtime/pyrho.json"

arms=(bottleneck_pyrho_constant bottleneck_pyrho_matched expansion_pyrho_constant expansion_pyrho_matched)
for arm in "${arms[@]}"; do
  rm -f "${DEMO_ROOT}/arms/${arm}/pyrho_table.hdf" \
    "${DEMO_ROOT}/arms/${arm}"/region_*.rmap \
    "${DEMO_ROOT}/arms/${arm}/pred_pyrho.npz" \
    "${DEMO_ROOT}/results/${arm}.json"
done
