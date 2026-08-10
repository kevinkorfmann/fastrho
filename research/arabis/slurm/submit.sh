#!/usr/bin/env bash
set -euo pipefail

SLURM_BIN=${SLURM_BIN:-/cm/local/apps/slurm/24.11/bin}
export PATH="${SLURM_BIN}:${PATH}"
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
ROOT=${ARABIS_ROOT:-/vast/projects/smathi/cohort/${USER}/fastrho-arabis}
mkdir -p "${ROOT}/logs"
cd "${REPO}"
exports="ALL,ARABIS_REPO=${REPO},ARABIS_ROOT=${ROOT}"

prep=$(sbatch --parsable --export="${exports}" research/arabis/slurm/00_prepare.sbatch)
prep=${prep%%;*}
python_env=$(sbatch --parsable --dependency="afterok:${prep}" --export="${exports}" research/arabis/slurm/00_inference_env.sbatch)
python_env=${python_env%%;*}
fetch=$(sbatch --parsable --dependency="afterok:${prep}" --array=0-36%20 --export="${exports}" research/arabis/slurm/01_download.sbatch)
fetch=${fetch%%;*}
# Pair each alignment task with the correspondingly indexed completed download,
# allowing Betty to overlap I/O and CPU work without exposing partial FASTQs.
align=$(sbatch --parsable --dependency="aftercorr:${fetch}" --array=0-36%37 --export="${exports}" research/arabis/slurm/02_align.sbatch)
align=${align%%;*}
call=$(sbatch --parsable --dependency="afterok:${align}" --array=0-53%54 --export="${exports}" research/arabis/slurm/03_call.sbatch)
call=${call%%;*}
finalize=$(sbatch --parsable --dependency="afterok:${call}" --export="${exports}" research/arabis/slurm/04_finalize.sbatch)
finalize=${finalize%%;*}
infer=$(sbatch --parsable --dependency="afterok:${finalize}:${python_env}" --export="${exports}" research/arabis/slurm/05_infer.sbatch)
infer=${infer%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:${infer}" --export="${exports}" research/arabis/slurm/06_evaluate.sbatch)
evaluate=${evaluate%%;*}

printf 'prepare=%s\npython_env=%s\nfetch=%s\nalign=%s\ncall=%s\nfinalize=%s\ninfer=%s\nevaluate=%s\n' \
  "${prep}" "${python_env}" "${fetch}" "${align}" "${call}" "${finalize}" "${infer}" "${evaluate}" |
  tee "${ROOT}/submission.tsv"
