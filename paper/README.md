# Paper analysis layout

The authoritative submission, **Scalable inference of recombination maps across evolutionary
contexts**, lives in
[`kevinkorfmann/fastrho-manuscript-2026-07-21`](https://github.com/kevinkorfmann/fastrho-manuscript-2026-07-21).
Its active sources are the neutral `main.tex` and `si.tex`; phase-specific variants are legacy.

This directory contains the paper's analysis products and provenance:

- `manuscript/figures/` and `manuscript/generated/` — package outputs copied into a disposable
  staged manuscript tree;
- `figures/`, `tables/`, `figdata/`, and `results_snapshot/` — committed figure and number inputs;
- `data_provenance.yaml` and `figure_provenance.json` — external-data and figure ledgers;
- `anopheles_variants/ag3/` — the active 13-population mosquito atlas and released maps;
- `anopheles_variants/phase2/` — superseded open-data analysis retained for historical verification;
- `legacy/` — superseded public material, never an active input.

The public map bundles can be rebuilt from the repository root with:

```bash
python scripts/export_paper_data.py
```

The manuscript repository remains authoritative for prose and layout. The reproducibility hub holds
analysis producers, checkpoint metadata, released maps, and historical frozen workflow records; it
does not make files under `legacy/` active again.
