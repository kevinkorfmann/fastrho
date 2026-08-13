# Paper analysis layout

The authoritative submission, **Scalable inference of recombination maps across evolutionary
contexts**, lives in
[`kevinkorfmann/fastrho-manuscript-2026-07-21`](https://github.com/kevinkorfmann/fastrho-manuscript-2026-07-21).
Only its Phase 2 main manuscript and SI are active.

This directory contains the paper's analysis products and provenance:

- `manuscript/figures/` and `manuscript/generated/` — package outputs copied into a disposable
  staged manuscript tree;
- `figures/`, `tables/`, `figdata/`, and `results_snapshot/` — committed figure and number inputs;
- `data_provenance.yaml` and `figure_provenance.json` — external-data and figure ledgers;
- `anopheles_variants/phase2/` — the active open-data mosquito analysis;
- `legacy/` — superseded public material, never an active input.

Rebuild and verify the Phase 2 article and SI from the repository root:

```bash
./reproduce/run.sh
```

For a PDF-only compile after artifacts have been staged, use
`uv run python reproduce/build_manuscript.py`.

The build writes the locked manuscript snapshot under `tmp/reproduce/manuscript/` and final PDFs
under `output/pdf/`. It never edits the authoritative manuscript checkout.

The ordered rebuild commands and active producer list are in [`../scripts/README.md`](../scripts/README.md).

The exact authority boundary is machine-readable in [`../reproduce/manuscript.lock.json`](../reproduce/manuscript.lock.json).
