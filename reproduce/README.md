# Reproduce the Phase 2 paper

This is the entry point for the active main manuscript and SI. The manuscript itself remains in
[`kevinkorfmann/fastrho-manuscript-2026-07-21`](https://github.com/kevinkorfmann/fastrho-manuscript-2026-07-21);
`manuscript.lock.json` pins its exact Phase 2 commit and files. Phase 3 and earlier variants are
legacy and are not part of this workflow.

```bash
./reproduce/run.sh
```

The command fetches the locked manuscript into `tmp/reproduce/manuscript`, regenerates analysis
artifacts, stages them there, audits the result, and compiles both PDFs. It never writes to the
manuscript checkout.

## Find the code behind a result

| Paper result | Primary producer | Upstream record |
|---|---|---|
| Simulation qualification | `scripts/fig_manuscript.py` | `paper/results_snapshot/summary.json` |
| Canid and selfing history | `scripts/fig_manuscript_history.py` | `paper/analysis/canid/`, `research/demography_matched/`, `research/arabis/` |
| Phase 2 atlas, 2La, crosses, resistance, pyrho | `paper/anopheles_variants/common/` | `paper/anopheles_variants/phase2/` |
| Redpoll arrangements | `scripts/fig_redpoll_karyotype.py` | `paper/figdata/redpoll_*` |
| Gene conversion | `scripts/fig_gene_conversion.py` | `paper/results_snapshot/gene_conversion.json` |
| Unphased inputs and method limits | `scripts/fig_unphased.py`, `scripts/fig_si_unique.py` | `paper/figdata/`, `paper/results_snapshot/` |
| Ten-species comparison | `scripts/fig_treeoflife_panel.py` | `paper/figdata/transect.json` |
| *Arabis* cross comparison | `scripts/fig_arabis_cross.py` | `research/arabis/`, `paper/results_snapshot/arabis_*` |

`artifacts.json` gives the exact mapping from every one of the 15 included PDFs and seven generated
TeX/BibTeX files to its package output and producer. `paper/data_provenance.yaml` records public raw
sources and preprocessing scripts; `paper/figure_provenance.json` records figure inputs, commands,
and hashes.

## Useful checks

```bash
# Fetch and verify only the locked manuscript snapshot.
uv run python reproduce/fetch_manuscript.py

# Show every command without running it.
uv run python reproduce/paper.py plan --profile paper

# Verify model archives and paper metadata without rerendering figures.
uv run python reproduce/paper.py run --profile verify

# Print the machine-readable producer inventory.
uv run python reproduce/paper.py inventory
```

Large checkpoints are GitHub release assets, not Git objects. User-facing model selection and
verified downloads are documented in [`docs/checkpoints.md`](../docs/checkpoints.md). Frozen
paper-only ensembles are kept in the `paper-phase2-checkpoints-v1` support release and are clearly
separated from checkpoints intended for new analyses. These inference-only copies preserve the
exact trained weights and loading metadata; `checkpoints.json` also records each original training
checkpoint hash.

```bash
uv run python reproduce/fetch_support_checkpoints.py --group arabis-structured-ensemble
```

Simulation and training entry points for every model named in the paper are indexed in
[`models/TRAINING.md`](../models/TRAINING.md). They use repository-relative paths and caller-supplied
output locations, so retraining does not depend on the original Betty or Sesame filesystems.
