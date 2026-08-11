#!/bin/bash
# Re-extract every EVA species from its cached VCF using multi-contig pooling (rescues scaffold-level
# assemblies that failed the one-contig window gate), then re-infer + re-QC + re-aggregate. The
# render-loop picks up the new transect.json. Runs on sesame (hours).
D=/home/kkor/realdata; PY=/home/kkor/venvs/fastrho/bin/python
: > "$D/reextract.log"
for cfg in "$D/master_eva.cfg" "$D/master_eva2.cfg"; do
  [ -s "$cfg" ] || continue
  while IFS="|" read -r key url chrom start end mode mu thin maxs sregex mapsp mapid maxmiss indexed; do
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    vcf="$D/ext/$key.vcf.gz"
    [ -s "$vcf" ] || { echo "[$key] no cached vcf" >> "$D/reextract.log"; continue; }
    [ -z "$maxmiss" ] && maxmiss=0.3; [ -z "$maxs" ] && maxs=0
    rargs=""; [ "$sregex" != "-" ] && [ -n "$sregex" ] && rargs="--sample-regex $sregex"
    cd /home/kkor/fastrho
    PYTHONNOUSERSITE=1 $PY scripts/transect_extract.py --vcf "$vcf" --top-contigs 12 --mode "$mode" \
      --mu "$mu" --thin 3 --max-samples "$maxs" --max-missing "$maxmiss" $rargs \
      --out "$D/hap/$key.npz" 2>&1 | grep "^\[extract\] [0-9]" | tail -1 | sed "s/^/[$key] /" >> "$D/reextract.log"
    rm -f "$D/transect_$key.json" "$D/qc_$key.json"
  done < "$cfg"
done
echo "REEXTRACT_DONE $(date +%T)" >> "$D/reextract.log"
cd /home/kkor/fastrho_dr
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 DEV=cuda:0 $PY scripts/transect_infer_all.py "$D/hap" "$D" 2>/dev/null | grep -c "^INFER" | sed 's/^/inferred: /' >> "$D/reextract.log"
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 DEV=cuda:0 $PY scripts/transect_qc_all.py "$D/hap" "$D" 2>/dev/null | grep -c "^QC" | sed 's/^/qcd: /' >> "$D/reextract.log"
$PY scripts/build_transect_json.py "$D" "$D/transect.json" 2>&1 | grep -E "KEPT" >> "$D/reextract.log"
echo "POOLED_ALL_DONE $(date +%T)" >> "$D/reextract.log"
