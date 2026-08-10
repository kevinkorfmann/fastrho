"""Options for verification tests that intentionally contact external services."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-online",
        action="store_true",
        default=False,
        help="verify live dataset URLs and DOI metadata against remote services",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-online"):
        return
    marker = pytest.mark.skip(reason="requires --run-online and external network access")
    for item in items:
        if "online" in item.keywords:
            item.add_marker(marker)
