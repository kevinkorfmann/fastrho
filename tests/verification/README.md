# Cross-domain manuscript verification

This directory audits the active manuscript, its bibliography, its declared
data and code, and the torch-free implementation primitives. It complements the
claim re-derivation suite in `tests/paper/` and the package unit tests in
`tests/test_*.py`.

## Audit domains

| Domain | Contract |
|---|---|
| Quantitative claims | Every occurrence of every reader-facing numeric literal in the active main text, SI, and included tables resolves to an exact scoped source value, with explicit rounding and percent/kb/Mb conversions. Repeated values are retained as separate audit records. Central results are also independently re-derived by `tests/paper/`. |
| References | Every BibTeX record has complete type-appropriate metadata; every active citation resolves; DOI/URL syntax, duplicate active works, placeholder text, and implausible years are rejected. |
| Live citation accuracy | DOI-backed references are compared with Crossref titles and publication years. Temporary Crossref rate limits are reported as skips, not false citation failures. |
| Resources | Every provenance-ledger record has a stable source, terms route, citations, derivatives, and producing scripts. Every declared local artifact must exist and be nonempty. Live endpoints are checked separately. |
| Result artifacts | Every committed JSON result is strict JSON with finite numeric values. Every NPZ is readable without pickle, contains no object arrays or infinities, and is not empty. Missing-value NaNs are permitted because several aligned map arrays intentionally encode uncovered boundary windows. |
| Figures and tables | Every included graphic has one executable producer plus checksummed inputs and output; every TeX fragment exists; PDFs/PNGs have valid signatures; labels and references resolve; every display has a caption and namespaced label. |
| Code | Every package module and every provenance-producing Python script parses as Python; producing shell scripts pass `bash -n`; deterministic properties exercise filtering and Gaussian chunk stitching. |
| Submission | Abstract/significance limits, manuscript/SI structure, source disclosure, checkpoint status, and journal-facing package contracts are checked by this suite and `tests/paper/test_manuscript_submission.py`. |

## Commands

```bash
# Entire deterministic suite
uv run --extra dev python -m pytest -q

# Only the cross-domain layer
uv run --extra dev python -m pytest tests/verification -q

# Live resource and DOI audit
uv run --extra dev python -m pytest tests/verification/test_online_resources.py \
  --run-online -m online -q
```

The live audit is intentionally opt-in because repository availability, publisher
rate limits, and institutional firewalls are external state. A deterministic pass
proves internal consistency and reproducibility contracts; it does not by itself
prove that an observational interpretation is biologically true. That stronger
standard still requires the committed raw evidence, the re-derivation tests,
independent validation described in the manuscript, and scientific peer review.

The generated `paper/reproducibility_audit.json` is the machine-readable handoff: it lists every
printed numeric occurrence with its line, column, exact source locator, and conversion route, and
every included figure with its producer, command, input hashes, and output hash.

## Adding material

New bibliography entries, provenance datasets, result JSON/NPZ files, figures,
included TeX tables, producing scripts, cross-references, and numeric passages are
automatically parameterized into new cases. Do not add a manual count-padding test.
Each new case must correspond to a distinct claim, source, artifact, or invariant.
