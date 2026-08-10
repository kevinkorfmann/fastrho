#!/bin/bash
# Does pyrho still invert the A. thaliana chr1 map under a REAL Arabidopsis demography?
# Rebuild the pyrho lookup table under each stdpopsim AraTha model, holding EVERYTHING else
# identical to the paper run (same chr1 region VCFs, same optimize flags), then re-score.
set -u
RD=/home/kkor/realdata
PYBIN=/home/kkor/venvs/pyrho/bin/pyrho
FR=/home/kkor/venvs/fastrho/bin/python
SRC=$RD/pyrho/athal_c1                     # existing chr1 region VCFs (paper run)
source $RD/demog_params.sh
LOG=$RD/demog_check.log
exec >>"$LOG" 2>&1
echo "===== START $(date) ====="

run_one () {
  name=$1; P=$2; T=$3
  d=$RD/pyrho/athal_c1_$name
  mkdir -p "$d"; rm -f "$d"/region_*.rmap "$d"/region_*.vcf
  cp $SRC/region_*.vcf "$d"/
  tbl=$d/table.hdf
  echo "--- $name: make_table (epochs in -p) ---"
  if [ -n "$T" ]; then
    $PYBIN make_table -n 156 -m 7e-09 -p "$P" -t "$T" --approx -N 161 --numthreads 24 -o "$tbl" \
      2>&1 | tail -2
  else
    $PYBIN make_table -n 156 -m 7e-09 -p "$P" --approx -N 161 --numthreads 24 -o "$tbl" \
      2>&1 | tail -2
  fi
  n=0
  for v in "$d"/region_*.vcf; do
    b=${v%.vcf}
    $PYBIN optimize --vcffile "$v" --tablefile "$tbl" --ploidy 1 -w 50 -bpen 50 \
      --numthreads 8 -o "$b.rmap" 2>/dev/null && n=$((n+1))
  done
  echo "  $name: optimized $n regions"
  echo -n "SCORE[$name] "
  $FR $RD/score_rmaps.py "$d" 1 $RD/hap/athal_c1.npz
  rm -f "$tbl"    # tables are large; drop after scoring
}

echo ">>> baseline (paper): constant Ne=266722 gave Pearson=-0.355 on chr1"
run_one af2  "$AF2_P"  "$AF2_T"    # African2Epoch (expansion), 2 epochs -- fastest
run_one af3  "$AF3_P"  "$AF3_T"    # African3Epoch (strong bottleneck), 3 epochs
run_one smac "$SMAC_P" "$SMAC_T"   # SouthMiddleAtlas coarsened to 6 epochs (piecewise-constant shape)

touch $RD/DEMOG_CHECK_DONE
echo "===== DONE $(date) ====="
