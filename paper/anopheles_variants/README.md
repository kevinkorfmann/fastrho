# Active *Anopheles* manuscript analysis

This directory contains the mosquito analyses associated with the manuscript. The active atlas uses
MalariaGEN Ag3.0 phased haplotypes.

- `ag3/` contains 65 inferred maps for 13 populations across five chromosome arms and the public
  BED release.
- `phase2/` and `common/` retain the superseded open-data analysis and its release-specific tools
  for historical verification only; they are not active manuscript inputs.

Large genotype inputs remain outside Git. The fixed Ag3 cohort manifest, committed maps, and
ready-to-use downloads are documented in [`docs/data.md`](../../docs/data.md). Regenerate the public
tables and bundles with `python scripts/export_paper_data.py`.
