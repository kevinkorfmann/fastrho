#!/usr/bin/env bash
set -euo pipefail

phase2_root="${1:-/home/kkor/agam_phase2}"
base="ftp://ngs.sanger.ac.uk/production/ag1000g/phase2/AR1"
mkdir -p "$phase2_root/raw/haplotypes_main_hdf5" \
  "$phase2_root/raw/haplotypes_crosses_shapeit" \
  "$phase2_root/raw/metadata" \
  "$phase2_root/provenance"

for arm in 2R 2L 3R 3L X; do
  wget -c -P "$phase2_root/raw/haplotypes_main_hdf5" \
    "$base/haplotypes/main/hdf5/ag1000g.phase2.ar1.haplotypes.${arm}.h5"
done

for arm in 2R 2L 3R 3L; do
  wget -c -P "$phase2_root/raw/haplotypes_crosses_shapeit" \
    "$base/haplotypes/crosses/shapeit/ag1000g.phase2.ar1.haplotypes.${arm}.gz"
  wget -c -P "$phase2_root/raw/haplotypes_crosses_shapeit" \
    "$base/haplotypes/crosses/shapeit/ag1000g.phase2.ar1.samples.${arm}.gz"
done

for name in samples.meta.txt samples.species.txt cross.samples.meta.txt samples.kdr.txt samples.rdl.txt; do
  wget -c -P "$phase2_root/raw/metadata" "$base/samples/$name"
done
for name in haplotypes.autosomes.meta.txt haplotypes.X.meta.txt; do
  wget -c -P "$phase2_root/raw/metadata" "$base/haplotypes/main/$name"
done

find "$phase2_root/raw" -type f -print0 \
  | sort -z \
  | xargs -0 shasum -a 256 \
  > "$phase2_root/provenance/raw_files.sha256"
date -u +%Y-%m-%dT%H:%M:%SZ > "$phase2_root/provenance/download_completed_utc.txt"
echo "$phase2_root/provenance/raw_files.sha256"
