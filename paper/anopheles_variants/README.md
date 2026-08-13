# Active *Anopheles* manuscript analysis

This directory contains the mosquito analysis used by the current manuscript. The active dataset is
the freely available Ag1000G Phase 2 AR1 release; no Ag3/Phase 3 result is a current manuscript input.

- `phase2/` contains the frozen nine-population design, 45 inferred maps, compact result files,
  figures, manuscript fragments, provenance manifests, and the public BED release.
- `common/` contains the Phase 2 extraction, inference, statistics, plotting, and verification utilities.

Restricted Phase 3 analyses, results, maps, and manuscript text are not distributed in the public
repository. Their private preservation is separate from this Phase 2 release.

Large public genotype inputs remain outside Git. Phase 2 records source URLs, selected samples,
commands, environments, and checksums for every promoted artifact. The ready-to-use atlas and result
tables are documented in `docs/data.md`.

The checksum-bound Phase 2 outputs are committed and do not require access to the original compute
host. Run the repository-level workflow to verify and stage them into the locked manuscript snapshot:

```bash
./reproduce/run.sh
```

The Phase 2 gate verifies the 45 maps, promoted results, fragments, provenance structure, and absence
of restricted-release claims.
