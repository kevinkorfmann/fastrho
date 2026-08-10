#!/bin/bash
# Build a C. remanei (outcrossing nematode) population VCF from 14 wild-isolate WGS read sets
# (Teterina/Cutter), then it can be extracted+inferred like any transect species. Runs on sesame.
set -o pipefail
D=/home/kkor/realdata; CR=$D/cr; CONDA=/home/kkor/miniconda3
source "$CONDA/etc/profile.d/conda.sh"
RUNS="SRR17772302 SRR17772303 SRR17772304 SRR17772305 SRR17772306 SRR17772307 SRR17772308 SRR17772309 SRR17772310 SRR17772311 SRR17772312 SRR17772313 SRR17772314 SRR17772315"
log(){ echo "[cr $(date +%T)] $*" >> "$D/cr_build.log"; }
: > "$D/cr_build.log"
mkdir -p "$CR"; cd "$CR" || exit 1
# wait for conda env tools + reference
for i in $(seq 1 90); do
  conda activate cr 2>/dev/null
  command -v minimap2 >/dev/null 2>&1 && command -v bcftools >/dev/null 2>&1 && command -v seqtk >/dev/null 2>&1 && [ -s "$CR/ref.status" ] && break
  sleep 20
done
conda activate cr
log "tools: $(command -v minimap2) $(command -v bcftools) $(command -v seqtk); ref $(ls -la ref.fa.gz 2>/dev/null|awk '{print $5}')"
gunzip -kf ref.fa.gz && log "unzipped ref" || { log "no ref"; exit 1; }
awk '/^>/{h=$0;p=(!(h in s));s[h]=1}p' ref.fa > ref_dd.fa && mv ref_dd.fa ref.fa   # drop duplicate contigs
rm -f ref.fa.fai ref.mmi
log "ref contigs after dedup: $(grep -c '>' ref.fa)"
minimap2 -d ref.mmi ref.fa 2>>"$D/cr_build.log"
BAMS=""
for r in $RUNS; do
  bam="$CR/$r.bam"
  if [ -s "$bam" ]; then BAMS="$BAMS $bam"; continue; fi
  # resolve ENA fastq urls
  urls=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=fastq_ftp&format=tsv" 2>/dev/null | tail -1 | cut -f2)
  f1=$(echo "$urls" | cut -d';' -f1); f2=$(echo "$urls" | cut -d';' -f2)
  [ -z "$f1" ] && { log "$r no fastq url"; continue; }
  log "$r downloading"
  curl -sL -o "${r}_1.fq.gz" "https://$f1"
  [ -n "$f2" ] && [ "$f2" != "$f1" ] && curl -sL -o "${r}_2.fq.gz" "https://$f2"
  # subsample to ~1.5M read pairs (~15x for a 130Mb genome), align, sort
  seqtk sample -s11 "${r}_1.fq.gz" 1500000 2>/dev/null | gzip > s1.fq.gz
  if [ -s "${r}_2.fq.gz" ]; then
    seqtk sample -s11 "${r}_2.fq.gz" 1500000 2>/dev/null | gzip > s2.fq.gz
    minimap2 -ax sr -t 8 -R "@RG\tID:$r\tSM:$r" ref.mmi s1.fq.gz s2.fq.gz 2>>"$D/cr_build.log" | samtools sort -@4 -o "$bam" - 2>>"$D/cr_build.log"
  else
    minimap2 -ax sr -t 8 -R "@RG\tID:$r\tSM:$r" ref.mmi s1.fq.gz 2>>"$D/cr_build.log" | samtools sort -@4 -o "$bam" - 2>>"$D/cr_build.log"
  fi
  samtools index "$bam" 2>>"$D/cr_build.log"
  rm -f "${r}_1.fq.gz" "${r}_2.fq.gz" s1.fq.gz s2.fq.gz
  n=$(samtools idxstats "$bam" 2>/dev/null | awk '{s+=$3} END{print s}')
  log "$r aligned ($n mapped reads)"
  BAMS="$BAMS $bam"
done
log "calling variants on $(echo $BAMS | wc -w) bams"
# largest contig only (fast + enough windows); pick it from the reference index
BIG=$(samtools faidx ref.fa 2>/dev/null && sort -k2,2 -nr ref.fa.fai | head -1 | cut -f1)
log "biggest contig=$BIG"
bcftools mpileup -f ref.fa -r "$BIG" -a AD -q 20 -Q 20 -Ou $BAMS 2>>"$D/cr_build.log" \
  | bcftools call -mv -Oz -o cr.vcf.gz 2>>"$D/cr_build.log"
bcftools index -t cr.vcf.gz 2>>"$D/cr_build.log"
ns=$(bcftools query -l cr.vcf.gz 2>/dev/null | wc -l); nv=$(bcftools index -n cr.vcf.gz 2>/dev/null)
log "cr.vcf.gz: $ns samples, $nv variants, contig $BIG"
echo "CR_VCF_DONE $BIG" > "$D/cr.status"
