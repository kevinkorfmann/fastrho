"""Every cell of the manuscript tables must equal the canonical snapshot value
rounded to the precision the table prints. This is the machine-checkable form of
the repro.tex promise that table cells are read from the results JSON, not typed.

Covered here:
  * Table 1 (main_results.tex)  -- Pearson @100kb / @25kb, fastrho/pyrho/ReLERNN
The pyrho head-to-head table is checked separately because it has a different data source.
"""

import json

import known_issues
import paperlib as P
import pytest

pytestmark = pytest.mark.consistency


# main_results.tex column order:
#   config | f@100 p@100 r@100 | f@25 p@25 r@25
_COLSPEC = [
    ("100kb", "fastrho"),
    ("100kb", "pyrho"),
    ("100kb", "relernn"),
    ("25kb", "fastrho"),
    ("25kb", "pyrho"),
    ("25kb", "relernn"),
]


def _main_rows():
    tex = P.read_tex("main_results.tex")
    rows = []
    for cells in P.tex_rows(tex):
        cfg = P.clean_cell(cells[0])
        if cfg not in P.BENCH_CONFIGS:
            continue  # skip the two header rows
        rows.append((cfg, cells[1:7]))
    return rows


MAIN_ROWS = _main_rows()


def test_main_results_table_parsed():
    # The table lists exactly these eight configurations.
    expected = {
        "const_n20",
        "const_n40",
        "real_hapmap",
        "real_decode",
        "bottleneck_n20",
        "expansion_n20",
        "real_dog",
        "const_n100",
    }
    assert {cfg for cfg, _ in MAIN_ROWS} == expected


def _cell_cases():
    cases = []
    for cfg, cells in MAIN_ROWS:
        for written, (scale, method) in zip(cells, _COLSPEC):
            cases.append((cfg, scale, method, written.strip()))
    return cases


CASES = _cell_cases()


@pytest.mark.parametrize(
    "config,scale,method,written", CASES, ids=[f"{c}-{s}-{m}" for c, s, m, _ in CASES]
)
def test_main_results_cell_matches_snapshot(config, scale, method, written):
    val = P.as_number(written)
    if val is None:
        # An em-dash / '--' cell: assert the snapshot genuinely lacks this cell
        # (or scored too few intervals to report a Pearson).
        cell = P.summary()[config]["scales"].get(scale, {}).get(method, {})
        assert "pearson" not in cell, (
            f"{config}/{scale}/{method}: table shows '--' but snapshot has "
            f"pearson={cell.get('pearson')}"
        )
        return
    # Table prints a number — the snapshot must have that cell.
    try:
        paired_path = P.REPO_ROOT / "paper" / "results_snapshot" / "demography_matched.json"
        if (
            paired_path.is_file()
            and method in {"pyrho", "relernn"}
            and config in {"bottleneck_n20", "expansion_n20"}
        ):
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            actual = paired["scenarios"][config.removesuffix("_n20")][method]["arms"]["matched"][
                scale
            ]["pearson"]
        else:
            actual = P.metric(config, scale, method, "pearson")
    except KeyError:
        key = f"table.{config}.{scale}.{method}"
        if key in known_issues.KNOWN:
            pytest.xfail(known_issues.KNOWN[key])
        pytest.fail(
            f"{config}/{scale}/{method}: table prints {written} but "
            f"results_snapshot/summary.json has no such cell"
        )
    assert P.matches_rounded(actual, written), (
        f"{config}/{scale}/{method}: table prints {written}, snapshot has "
        f"{actual:.6f} (rounds to {round(actual, len(written.split('.')[-1]))})"
    )
