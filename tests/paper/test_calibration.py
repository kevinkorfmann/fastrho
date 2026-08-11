"""Held-out calibration: the coverage curve, the 365,280-interval count, and the
specific nominal->empirical points the paper quotes, recomputed from the heldout
record.
"""

import paperlib as P
import pytest

pytestmark = pytest.mark.calibration


def test_n_held_out_intervals_is_365280():
    assert P.summary()["heldout"]["n_intervals"] == 365280


def test_coverage_curve_is_monotone_and_well_behaved():
    h = P.summary()["heldout"]["coverage_curve"]
    nom, emp = h["nominal"], h["empirical"]
    assert nom == sorted(nom) and emp == sorted(emp)
    assert len(nom) == len(emp) == 6
    # empirical coverage tracks nominal within a few points everywhere
    for n, e in zip(nom, emp):
        assert abs(e - n) < 0.04, f"coverage {e:.3f} far from nominal {n}"


@pytest.mark.parametrize("nominal,written", [
    (0.90, "0.913"),    # data.tex / repro.tex
    (0.95, "0.954"),    # data.tex / repro.tex
    (0.95, "95.4%"),    # abstract / main caption
    (0.90, "91%"),      # between_pop.tex
])
def test_quoted_coverage_points(nominal, written):
    h = P.summary()["heldout"]["coverage_curve"]
    emp = h["empirical"][h["nominal"].index(nominal)]
    assert P.matches_rounded(emp, written)


def test_nominal_range_is_05_to_099():
    nom = P.summary()["heldout"]["coverage_curve"]["nominal"]
    assert min(nom) == 0.5 and max(nom) == 0.99
