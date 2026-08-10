#!/usr/bin/env bash
#SBATCH --job-name=find-rel-native
#SBATCH --partition=genoa-std-mem
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=5632M
#SBATCH --output=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.out
#SBATCH --error=/vast/projects/smathi/cohort/%u/fastrho-demography-v1/logs/%x-%j.err
set -euo pipefail

# Read-only provenance recovery for the legacy native-window comparison.  The
# output is deliberately restricted to paths whose names identify ReLERNN or
# one of the four configurations printed in the SI table.
roots=(
  "/vast/projects/smathi/cohort/${USER}"
  "${HOME}"
)
for root in "${roots[@]}"; do
  [[ -d "${root}" ]] || continue
  find "${root}" -type f \
    \( -iname '*relernn*' -o -iname '*const_n20*' -o -iname '*const_n40*' \
       -o -iname '*real_decode*' -o -iname '*real_hapmap*' \
       -o -iname 'native_final.json' -o -iname 'native.json' \
       -o -path '*/results_final/*.json' \) \
    -not -path '*/.cache/*' -not -path '*/cache/*' -not -path '*/software/*' \
    -printf '%s\t%TY-%Tm-%TdT%TH:%TM:%TS\t%p\n'
done | sort -t $'\t' -k3,3 -u
