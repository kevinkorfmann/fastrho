#!/bin/bash
# Extended-Data selfer figure (fig_selfer_inversion): run A. thaliana chr1-5 through
# extract -> fastrho(self2) -> pyrho, on sesame. Reuses the chr1 pyrho lookup table
# (Ne is genome-wide) for every chromosome, so the whole-genome extension stays
# identical to the published chr1 case (Fig. realdata c). Then:
#     /home/kkor/venvs/fastrho/bin/python scripts/bundle_selfer_chroms.py
# writes paper/figdata/selfer_chroms.npz that scripts/fig_selfer_inversion.py renders.
# Helpers (selfer_chroms_extract.py, selfer_pyrho_fixcfg.py) are staged next to this
# script in /home/kkor/realdata; adjust EXTRACT/FIXCFG if you keep them elsewhere.
set -u
EXTRACT=/home/kkor/realdata/selfer_chroms_extract.py
FIXCFG=/home/kkor/realdata/selfer_pyrho_fixcfg.py
cd /home/kkor/fastrho
export PYTHONNOUSERSITE=1
AGAM=/home/kkor/venvs/agam/bin/python
FR=/home/kkor/venvs/fastrho/bin/python
PY=/home/kkor/venvs/pyrho/bin/python
RD=/home/kkor/realdata
TABLE=$RD/pyrho/athal/pyrho_table.hdf
POPSIZE=266722.42002369853
LOG=$RD/athal_ed.log

exec >>"$LOG" 2>&1
echo "================ START $(date) ================"

echo "--- [1/5] extract chr1-5 (agam venv) ---"
$AGAM "$EXTRACT" || { echo "EXTRACT_FAIL"; exit 1; }

echo "--- [2/5] fastrho self2 inference ---"
for c in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 $FR scripts/realdata_infer.py athal_c$c self2 2>&1 \
    | grep -E "athal_c|Error|Traceback" | grep -viE "FutureWarning|Warning"
done

echo "--- [3/5] pyrho setup + force genome-wide Ne + reuse table ---"
for c in 1 2 3 4 5; do
  $FR scripts/pyrho_rd.py athal_c$c setup
  d=$RD/pyrho/athal_c$c
  $FR "$FIXCFG" "$d/config.json" "$POPSIZE"
  ln -sf "$TABLE" "$d/pyrho_table.hdf"
done

echo "--- [4/5] pyrho optimize (5 chroms in parallel, 8 threads each) ---"
for c in 1 2 3 4 5; do
  $PY scripts/run_pyrho_config.py $RD/pyrho/athal_c$c > $RD/pyrho_athal_c$c.log 2>&1 &
done
wait
echo "pyrho optimize finished; per-chrom tails:"
for c in 1 2 3 4 5; do echo -n "  c$c: "; tail -1 $RD/pyrho_athal_c$c.log; done

echo "--- [5/5] pyrho score ---"
for c in 1 2 3 4 5; do
  $FR scripts/pyrho_rd.py athal_c$c score
done

touch $RD/ATHAL_ED_DONE
echo "================ DONE $(date) ================"
