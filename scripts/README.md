# Analysis scripts

User-facing checkpoint and map workflows are documented in
[`../docs/checkpoints.md`](../docs/checkpoints.md) and [`../docs/data.md`](../docs/data.md).
The data guide gives the command that regenerates the current public map tables and bundles.

[`../reproduce/`](../reproduce/) preserves the last fully pinned manuscript workflow. Run
`./reproduce/run.sh` only to reproduce that historical snapshot, or use
`uv run python reproduce/paper.py inventory` to list its registered dataset and figure producers.

The frozen snapshot is rebuilt through these ordered entry points. Run them from the repository
root; they use repository-relative paths and fail if a required committed input is missing.

## Component commands

| Order | Command | Output |
|---:|---|---|
| 1 | `uv run python scripts/build_manuscript_derived.py` | Compact derived quantities and the benchmark table |
| 2 | `uv run python scripts/export_paper_data.py` | Plot-ready public data downloads and checksums |
| 3 | `uv run --extra figures python scripts/build_manuscript_figures.py --run --write-manifest` | Every included figure and its input/output hashes |
| 4 | `uv run python reproduce/stage_manuscript.py` | Generated inputs staged into the locked historical snapshot |
| 5 | `uv run python reproduce/audit_phase2.py` | Number-to-source and artifact-to-producer verification |
| 6 | `uv run python reproduce/build_manuscript.py` | PDFs from the locked historical snapshot |
| 7 | `uv run python -m pytest tests/paper tests/verification -q` | Analysis, manuscript, and release contracts |
| 8 | `uv run python scripts/release_check.py --strict-models` | Public model, data, and provenance release gate |

`build_manuscript_figures.py` is the authoritative registry of active figure producers and their
direct inputs. `paper/data_provenance.yaml` is the authoritative registry of external datasets and
the scripts that created their committed derivatives. Together these avoid treating the many
exploratory scripts in this directory as part of the active rebuild.

## Demography-matched competitors

The constant-versus-matched ReLERNN and pyrho analysis has its own frozen, numbered Slurm workflow
under [`research/demography_matched/`](../research/demography_matched/). Its preparation, inference,
scoring, prediction archiving, and collation modules are:

1. `prepare_demography_benchmark.py`
2. `run_relernn_paired.py` and `run_pyrho_config.py`
3. `capture_demography_runtime.py`
4. `score_fastrho_reference.py`
5. `collate_demography_benchmark.py`
6. `archive_demography_predictions.py`

All cluster computation is submitted through the numbered Slurm files in that analysis directory.

## Model-release verification

Model publication is a separate release gate from rebuilding figures from committed arrays.
`fetch_model_release.py` downloads the registry archive, verifies its SHA-256, verifies every
member, and safely unpacks it. `package_model_release.py` performs the inverse release operation and
creates a byte-reproducible ZIP with `SHA256SUMS`. `verify_model_release.py` checks either loose
files or the ZIP. `select_model_checkpoint.py` chooses from the exact Lightning metrics CSV rather
than file modification times. The primary model's full Slurm workflow is under
`models/domain-randomized-v1/reproduce/`. Run `release_check.py --strict-models` before submission;
it fails unless every required paper model has a complete public release record.
Run `audit_public_releases.py` when online to compare all six user-model archives, checkpoints, and
statistics files—plus every paper-support asset—against the live GitHub asset names, byte sizes, and
SHA-256 digests without downloading the large weight files.

The frozen Arabis ensemble members are published in the paper-support model release. Its tag retains
the historical Phase 2 name. Submit
`slurm/audit_model_artifacts.sbatch` with their frozen JSON manifests to verify every checkpoint,
statistics archive, and preregistration file on a compute node before building a new deposit.

## Naming and status

- `fig_*.py` files render figures; only producers registered by `build_manuscript_figures.py` are
  active manuscript dependencies.
- Species-prefixed files (`agam_`, `arabis_`, `dog_`, `redpoll_`, `transect_`, and `selfer_`) hold
  extraction, inference, and analysis code for those systems.
- `deprecated/` contains scripts that are not active inputs and are retained only for history.
- Large raw data, environments, model checkpoints, and scheduler logs do not belong in Git. Their
  stable accessions or creation records belong in the provenance registry.

New manuscript scripts should accept paths as command-line arguments or environment variables,
derive the repository root from `__file__`, and write deterministic committed artifacts. Do not add
personal absolute paths to active paper entry points.
