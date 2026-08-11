"""pytest configuration for the paper-number re-derivation suite.

Makes ``paperlib`` importable from the sibling test modules without packaging,
and registers markers so the suite can be sliced (``-m rederive`` etc.).
"""

import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def pytest_configure(config):
    for line in (
        "consistency: prose/table number must match the canonical results JSON",
        "rederive: snapshot metric recomputed from committed raw arrays",
        "derived: a quantity computed from raw metrics (median/speedup/count/etc.)",
        "calibration: held-out coverage / CI calibration checks",
        "integrity: structural sanity of the results JSON itself",
    ):
        config.addinivalue_line("markers", line)


@pytest.fixture(scope="session")
def S():
    """The master summary.json record."""
    import paperlib
    return paperlib.summary()
