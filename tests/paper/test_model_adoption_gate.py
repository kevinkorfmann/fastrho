"""Adoption gates: a deployed model must clear its REAL-DATA bar, not just a simulated one.

Root cause of the 2026-07-04 regression
(see research/archive/model-zoo-15k-regression-postmortem.md): the 15k
selfing model was selected on simulated val_pearson (0.963, best in the whole zoo) but its real
A. thaliana recovery collapsed 0.27 -> 0.11. Nothing in CI looked at real-data recovery, so the
regressed model shipped into ED Fig 11. These tests are the missing gate: whatever selfing model
is deployed (whichever checkpoint bundle_selfer_chroms.py used to write selfer_chroms.npz) must

  (1) beat pyrho on every chromosome (the sign flip is the whole point of the figure),
  (2) recover a positive map genome-wide, and
  (3) clear a real-recovery FLOOR that the overfit 15k model (mean 0.11) would have failed
      but the deployed self2 model (mean 0.27) clears.

If a future retrain regresses selfing recovery and someone re-bundles the npz, these fail loudly
instead of silently re-rendering a broken figure.
"""

import numpy as np
import paperlib as P
import pytest

pytestmark = pytest.mark.derived

CHROMS = ("1", "2", "3", "4", "5")

# Floor on the deployed selfing model's genome-wide mean real recovery. Set to sit ABOVE the
# overfit-15k regression (0.11) and BELOW the good self2 model (0.27), so the gate catches a
# repeat of exactly this failure. Raise it if a better selfing model is adopted.
SELFER_REAL_RECOVERY_FLOOR = 0.20


def _selfer(kind):
    z = P.load_npz(P.FIGDATA / "selfer_chroms.npz")
    key = "_r" if kind == "fastrho" else "_pyrho_r"
    return np.array([float(z[f"c{c}{key}"]) for c in CHROMS])


def test_deployed_selfer_beats_pyrho_on_every_chromosome():
    """The deployed model must recover (r>0) where pyrho inverts (r<0), on all five chroms."""
    fr, py = _selfer("fastrho"), _selfer("pyrho")
    assert (fr > py).all(), f"fastrho does not beat pyrho on every chrom: fastrho={fr}, pyrho={py}"
    assert (fr > 0).all(), f"deployed selfer model not positive on every chrom: {fr}"
    assert (py < 0).all(), f"pyrho not inverted on every chrom (figure premise broken): {py}"


def test_deployed_selfer_clears_real_recovery_floor():
    """Genome-wide mean real recovery must clear the floor (catches the 0.11 overfit regression)."""
    mean_r = float(_selfer("fastrho").mean())
    assert mean_r >= SELFER_REAL_RECOVERY_FLOOR, (
        f"deployed selfing model mean real recovery {mean_r:.3f} < floor "
        f"{SELFER_REAL_RECOVERY_FLOOR} -- looks like a simulator-overfit checkpoint was adopted "
        f"(cf. 15k self=0.11 vs self2=0.27). Select the checkpoint on real data "
        f"(scripts/self_epoch_select.py), not on simulated val_pearson."
    )


def test_deployed_selfer_margin_over_pyrho_is_material():
    """The mean sign-flip margin (fastrho - pyrho) must be clearly positive, not a coin-flip."""
    margin = float(_selfer("fastrho").mean() - _selfer("pyrho").mean())
    assert margin >= 0.30, f"fastrho-vs-pyrho mean margin only {margin:.3f}; sign flip not robust"
