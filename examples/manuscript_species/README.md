# Manuscript species examples

This directory turns the empirical datasets in **Scalable inference of recombination maps across
evolutionary contexts** into small, inspectable presets. The active comparative SI panel is the
fixed set of ten species listed below. Dedicated presets additionally cover the dog-wolf analysis,
the three Ag3.0 *Anopheles* species, and redpoll.

The raw data are deliberately not committed: several sources are many gigabytes, have their own
terms, and change independently of fastrho. [`species.json`](species.json) records the source route,
cohort, contig, mutation rate, genotype view, model profile, and reporting scale. Downloads and maps
stay in ignored `data/` and `maps/` directories.

## The three-command workflow

Inspect a preset before downloading anything:

```bash
python3 examples/manuscript_species/data.py list
python3 examples/manuscript_species/data.py show human
python3 examples/manuscript_species/data.py download human --dry-run --include-companions
```

Download, select a biologically coherent cohort, and prepare a complete one-contig VCF. This human
example recreates the manuscript's CEU chromosome-2 slice:

```bash
python3 examples/manuscript_species/data.py download human --include-companions
awk '$2 == "CEU" {print $1}' \
  examples/manuscript_species/data/human.phase3.panel.tsv \
  > examples/manuscript_species/data/human.CEU.samples.txt
python3 examples/manuscript_species/data.py prepare human \
  --vcf examples/manuscript_species/data/human.chr2.vcf.gz \
  --samples examples/manuscript_species/data/human.CEU.samples.txt \
  --start 20000000 --end 36000000 \
  --out examples/manuscript_species/data/human.CEU.chr2.vcf.gz
```

Then run the preset with a compatible checkpoint and its matching feature-statistics archive:

```bash
python3 examples/manuscript_species/infer.py \
  --species human \
  --vcf examples/manuscript_species/data/human.CEU.chr2.vcf.gz \
  --checkpoint /path/to/domain-randomized.ckpt \
  --stats /path/to/feat_stats_dr.npz \
  --out examples/manuscript_species/maps/human.chr2.100kb.bed
```

Use `--dry-run` on `prepare` or `infer` to inspect the resolved command and scientific settings.
The model registry records the released checkpoint bundles and their supported input views.
Successful preparation and inference commands write adjacent provenance JSON files with the
resolved settings, sample IDs, and checkpoint/statistics hashes.

## Retained cross-species presets

| Key | Species | Paper use | Contig | $\mu$ | Model profile |
|---|---|---|---:|---:|---|
| `human` | *Homo sapiens* | core | 2 | 1.29e-8 | domain-randomized |
| `cattle` | *Bos taurus* × *B. indicus* | core | 1 | 1.2e-8 | domain-randomized |
| `sheep` | *Ovis aries* | core | 1 | 1.0e-8 | domain-randomized |
| `goat` | *Capra hircus* | core | 1 | 1.0e-8 | domain-randomized |
| `dmel` | *Drosophila melanogaster* | core | 2L | 5.5e-9 | domain-randomized |
| `athal` | *Arabidopsis thaliana* | core | 1 (score averaged over 1–5) | 7.0e-9 | selfing specialist |
| `donkey` | *Equus asinus* | context-limited | CM027690.1 | 7.2e-9 | domain-randomized |
| `jewelwasp` | *Nasonia vitripennis* | context-limited | NC_015868.2 | 3.0e-9 | domain-randomized |
| `aspen` | *Populus tremula* | core | chr1 | 1.0e-8 | domain-randomized |
| `chestnut` | *Castanea mollissima* | context-limited | Chr1 | 1.0e-8 | domain-randomized |

The manifest also retains rejected candidate cohorts with the role `excluded: cohort-design
review` so that earlier source evaluation remains reproducible; those entries are not part of the
cross-species figure or downloadable map set. The ten rows above—and only those ten rows—define the
active comparative SI panel. Dog, wolf, the three Ag3.0 *Anopheles* species, and redpoll remain
available under dedicated manuscript roles.

`data.py show KEY` is authoritative if this compact table and the manifest ever diverge.

## MalariaGEN Ag3.0

The active atlas uses MalariaGEN Ag3.0 phased haplotypes. Obtain the source data under the
MalariaGEN terms, then reproduce the frozen 40-mosquito panels listed in the atlas manifest:

```bash
cat paper/anopheles_variants/ag3/release/atlas_anopheles/manifest.tsv
python3 examples/manuscript_species/infer.py \
  --species anopheles_gambiae \
  --npz /path/to/ag3/extracted/gamb_BF__2R.npz \
  --checkpoint /path/to/high-ne.ckpt \
  --stats /path/to/high-ne-stats.npz \
  --out examples/manuscript_species/maps/gamb_BF.2R.50kb.bed
```

The active panel contains six *A. gambiae* populations (`gamb_BF`, `gamb_CM`, `gamb_GA`,
`gamb_GN`, `gamb_ML`, and `gamb_UG`), four *A. coluzzii* populations (`colu_BF`, `colu_CI`,
`colu_GM`, and `colu_ML`), and three *A. arabiensis* populations (`arab_MW`, `arab_TZ`, and
`arab_UG`). Repeat for arms `2R`, `2L`, `3R`, `3L`, and `X`. Every population has exactly 40
diploid mosquitoes, with the same individuals retained across all five arms. Ready-to-use BED maps
and the plot-ready combined table are linked from
[`docs/data.md`](../../docs/data.md).

## Redpoll and canids

Dryad serves the complete-call redpoll VCF through its interactive data-files panel, so the utility
prints the exact filename, size, and SHA-256 instead of embedding a brittle session URL:

```bash
python3 examples/manuscript_species/data.py show redpoll
python3 examples/manuscript_species/data.py prepare redpoll \
  --vcf /path/to/100p_0.05maf.vcf.gz \
  --out examples/manuscript_species/data/redpoll.chr1.vcf.gz
```

The pooled redpoll preset is sufficient for a population map. The manuscript's arrangement-specific
result additionally uses the PCA grouping and controls in `scripts/redpoll_karyotype_maps.py`.

Dog and wolf share the Plassais canid VCF. Download its metadata with `--include-companions`, create
separate sample-ID files for the intended dog and wolf cohorts, and pass each list to `data.py
prepare`. Never infer a mixed-species map accidentally.

## Adapt this to your species

The presets are worked examples, not a requirement that your organism appear here. For a new
dataset, copy the closest biological regime, then change the cohort, contig, mutation rate, and
checkpoint only when each change is justified. The general input contract and reporting checklist
are in [`docs/your-data.md`](../../docs/your-data.md).
