#!/bin/bash
# When the C. remanei VCF is built, extract -> infer (unphased DR) -> QC -> add meta + silhouette.
D=/home/kkor/realdata; PY=/home/kkor/venvs/fastrho/bin/python
MB=/home/kkor/fastrho_dr/paper/figdata/transect_meta.json
for i in $(seq 1 400); do grep -q CR_VCF_DONE "$D/cr.status" 2>/dev/null && break; sleep 60; done
BIG=$(awk '{print $2}' "$D/cr.status" 2>/dev/null)
[ -z "$BIG" ] && exit 1
cd /home/kkor/fastrho
PYTHONNOUSERSITE=1 $PY scripts/transect_extract.py --vcf "$D/cr/cr.vcf.gz" --chrom "$BIG" --start 1 \
  --end 999999999 --mode dosage --mu 3e-9 --thin 1 --max-missing 0.5 --out "$D/hap/remanei.npz" >> "$D/cr_build.log" 2>&1
cd /home/kkor/fastrho_dr
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 DEV=cuda:0 $PY scripts/transect_infer.py "$D/hap/remanei.npz" "$D/transect_remanei.json" >> "$D/cr_build.log" 2>&1
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 DEV=cuda:0 $PY scripts/transect_qc_all.py "$D/hap" "$D" remanei >> "$D/cr_build.log" 2>&1
$PY scripts/merge_meta.py "$MB" paper/figdata/transect_meta_cr.json >> "$D/cr_build.log" 2>&1
$PY scripts/phylopic_fetch.py paper/figdata/transect_meta_cr.json paper/figdata/silhouettes >> "$D/cr_build.log" 2>&1
echo "CR_INGEST_DONE $(date +%T)" >> "$D/cr_build.log"
