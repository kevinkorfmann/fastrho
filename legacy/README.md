# Legacy material

This directory preserves material that is not an input to the authoritative Phase 2 publication or
the importable `fastrho` package. Active code, data, figures, tests, and provenance remain outside
this directory and are registered by [`../reproduce/`](../reproduce/).

| Path | Preserved material |
|---|---|
| `pre-phase2-snapshot/` | Local, Git-ignored copy of files removed while establishing Phase 2 authority, restored from parent commit `ef5b336` with their original paths intact |
| `paper/` | Superseded manuscript artifacts, inactive figure outputs, Phase 3 work, and literature-review working files |
| `research/` | Ag3 pedigree/Phase 3 work, archived investigations, superseded provenance notes, and large-sample experiments |
| `scripts/deprecated/` | Scripts explicitly marked deprecated |
| `results/historical-root-results/` | Historical root-level result files that are not read by the Phase 2 workflow |
| `documentation/` | Superseded documentation and historical index stubs |
| `generated/` | Local builds, caches, and scratch outputs retained during cleanup; ignored by Git |

Legacy files are retained for due diligence and historical traceability. They are not maintained,
not imported by the package, and must not be cited as Phase 2 results.
