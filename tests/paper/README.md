# Paper tests

These tests validate committed result snapshots, tables, figures, documentation, model metadata,
and reader-facing contracts for the current paper.

Manuscript-aware tests read `main.tex` and `si.tex` from
`FASTRHO_MANUSCRIPT_ROOT`. The reproduction runner sets that variable to the locked snapshot under
`tmp/reproduce/manuscript/`. When the snapshot is absent, only manuscript-aware modules skip; the
package and result re-derivation tests remain runnable.

The historical frozen-snapshot audit remains named `reproduce/audit_phase2.py`; current
documentation, model-release, and data-release contracts are tested directly. Superseded manuscript
variants and their old prose registries are not part of the active documentation suite.
