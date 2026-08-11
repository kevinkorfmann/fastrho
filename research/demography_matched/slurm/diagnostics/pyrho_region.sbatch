#!/usr/bin/env bash
#SBATCH --job-name=demo-pyrho-debug
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/research/demography_matched/slurm/common.sh"

: "${PYRHO_ARM:?Set PYRHO_ARM to a prepared pyrho arm}"
: "${PYRHO_REGION:?Set PYRHO_REGION to a three-digit region number}"
config="${DEMO_ROOT}/arms/${PYRHO_ARM}"
mkdir -p "${DEMO_ROOT}/diagnostics"
"${DEMO_PYRHO_ENV}/bin/python" - "${config}/region_${PYRHO_REGION}.vcf" <<'PY'
import sys
import numpy as np
from pyrho.haplotype_reader import genos_to_configs, parse_vcf_to_genos

genos, positions = parse_vcf_to_genos(sys.argv[1], ploidy=1)
configs, _ = genos_to_configs(genos, window_size=50, ploidy=1)
n00, n01, n10, n11 = configs[:, 0], configs[:, 1], configs[:, 3], configs[:, 4]
n = n00 + n01 + n10 + n11
bad = (n00 + n01 == n) | (n10 + n11 == n) | (n01 + n11 == n) | (n10 + n00 == n)
print("parsed", genos.shape, "pairs", configs.shape[0], "missing", int((genos < 0).sum()))
print("nonsegregating_configs", int(bad.sum()))
if bad.any():
    indices = np.flatnonzero(bad)[:10]
    print("first_bad_indices", indices.tolist())
    print("first_bad_configs", configs[indices].tolist())
PY
"${DEMO_PYRHO_ENV}/bin/pyrho" optimize \
  --vcffile "${config}/region_${PYRHO_REGION}.vcf" \
  --tablefile "${config}/pyrho_table.hdf" --ploidy 1 \
  -w 50 -bpen 25 --numthreads "${SLURM_CPUS_PER_TASK}" \
  -o "${DEMO_ROOT}/diagnostics/${PYRHO_ARM}_region_${PYRHO_REGION}.rmap"
