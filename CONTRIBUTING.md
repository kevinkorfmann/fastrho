# Contributing

Thank you for improving `fastrho`. Please open an issue before a large change so the scientific
target, validation strategy, and compatibility implications are clear before implementation.

## Development setup

```bash
python -m pip install -e '.[sim,io,dev]'
```

Keep pull requests focused and include a test for changed behavior.

## Local checks

```bash
python -m pytest -q
ruff check fastrho tests scripts
python scripts/release_check.py
python -m build
```

## Repository organization

- Put package code under `fastrho/`, user-facing guidance under `docs/`, tests under `tests/`, and
  model-release utilities under `scripts/`.
- Keep generated PDFs, caches, downloaded weights, local datasets, and scratch outputs out of Git.
- Do not commit credentials, host-specific paths, private genotype data, or restricted datasets.
- Include enough provenance for a synthetic or openly licensed test fixture to be independently
  understood and regenerated.

Public contributions must concern the software or its public documentation. Research analyses and
unreleased article materials are maintained separately.
