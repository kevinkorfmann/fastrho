# Research workspace

This directory holds investigation code and records that support the package and manuscript but are
not part of the public API.

## Layout

| Path | Contents |
|---|---|
| [`demography_matched/`](demography_matched/) | Frozen paired ReLERNN/pyrho design, ordered Slurm jobs, environment pins, and reproduction guide |
| [`arabis/`](arabis/) | Frozen preprocessing, Slurm, model-selection, and evaluation workflow for the Arabis cross analysis |

Executable analysis and figure-generation code lives under [`scripts/`](../scripts/).

Only workflows supporting current package or manuscript results belong in this active tree.
Superseded investigations are preserved under [`../legacy/research/`](../legacy/research/).
