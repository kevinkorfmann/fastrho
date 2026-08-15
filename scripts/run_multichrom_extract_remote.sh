#!/usr/bin/env bash
set -euo pipefail

analysis_root=/home/kkor/fastrho_iain_20260814
analysis_python=/home/kkor/venvs/fastrho/bin/python
extractor=$analysis_root/code/multichrom_extract.py
output=$analysis_root/multichrom_hap
mkdir -p "$output"

"$analysis_python" "$extractor" \
  --vcf /home/kkor/realdata/dmel/DGRP2.dm6.SNPs.vcf.gz \
  --chrom 2L --chrom 2R --chrom 3L --chrom 3R --mode haploid \
  --max-missing 0.1 --min-maf 0.01 --thin 5 --mu 5.5e-9 \
  --map-sp DroMel --map-id ComeronCrossover_dm6 --prefix dmel --output "$output" \
  >"$analysis_root/logs/extract_dmel.log" 2>&1

"$analysis_python" "$extractor" \
  --vcf /home/kkor/realdata/ext/jewelwasp.vcf.gz \
  --chrom NC_015867.2 --chrom NC_015868.2 --chrom NC_015869.2 \
  --chrom NC_015870.2 --chrom NC_015871.2 --mode dosage \
  --max-missing 0.1 --min-maf 0.01 --thin 1 --mu 3e-9 \
  --prefix jewelwasp --output "$output" \
  >"$analysis_root/logs/extract_jewelwasp.log" 2>&1

"$analysis_python" "$extractor" \
  --vcf /home/kkor/realdata/ext/aspen.vcf.gz \
  --chrom chr1 --chrom chr2 --chrom chr3 --chrom chr4 --chrom chr5 --mode dosage \
  --max-missing 0.1 --min-maf 0.01 --thin 5 --mu 1e-8 \
  --prefix aspen --output "$output" \
  >"$analysis_root/logs/extract_aspen.log" 2>&1

touch "$analysis_root/MULTICHROM_EXTRACT_DONE"
