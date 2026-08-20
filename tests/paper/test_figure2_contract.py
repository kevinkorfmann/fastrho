"""Executable contract for every datum and label in main-text Figure 2.

The manuscript calls this Figure 2, while the historical output filename is
``fig1_method_validation.pdf`` and the renderer function is ``figure1``.  These
tests deliberately follow the reader-facing figure number and verify the live
Matplotlib artists, not just the plotting source or the committed PDF hash.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from manuscript_source import MAIN, SI
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = (
    ("Constant", "const_n20"),
    ("Bottleneck", "bottleneck_n20"),
    ("Expansion", "expansion_n20"),
    ("deCODE", "real_decode"),
    ("HapMap", "real_hapmap"),
    ("Dog", "real_dog"),
)
METHODS = ("fastrho", "pyrho", "relernn")
DISPLAY_NAMES = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}
METHOD_MARKERS = {"fastrho": "o", "pyrho": "s", "relernn": "^"}


def _load_script(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _caption() -> str:
    block = re.search(
        r"\\begin\{figure\*\}(?:(?!\\end\{figure\*\}).)*"
        r"\\label\{fig:qualification\}(?:(?!\\end\{figure\*\}).)*\\end\{figure\*\}",
        MAIN,
        flags=re.DOTALL,
    )
    assert block is not None
    caption = re.search(r"\\caption\{(.*)\}\s*\\label", block.group(0), flags=re.DOTALL)
    assert caption is not None
    return caption.group(1)


def _literal_assignments(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _expected(summary: dict, demography: dict, method: str, metric: str) -> np.ndarray:
    values = []
    for _label, key in SCENARIOS:
        if key in {"bottleneck_n20", "expansion_n20"}:
            scenario = key.removesuffix("_n20")
            if method == "fastrho":
                record = demography["scenarios"][scenario]["fastrho_reference"]["25kb"]
            else:
                record = demography["scenarios"][scenario][method]["arms"]["matched"]["25kb"]
        else:
            record = summary[key]["scales"]["25kb"][method]
        values.append(record[metric])
    return np.asarray(values, dtype=float)


def _open_expected(demography: dict, metric: str, *, panel: str) -> np.ndarray:
    offsets = {"pyrho": 0.0, "relernn": 0.22}
    x_shift = 0.09 if panel == "shape" else 0.07
    rows = []
    for method in ("pyrho", "relernn"):
        for scenario_index, scenario in enumerate(("bottleneck", "expansion"), start=1):
            value = demography["scenarios"][scenario][method]["arms"]["constant"]["25kb"]
            x = scenario_index - x_shift if panel == "shape" else scenario_index + offsets[method] - x_shift
            rows.append((x, value[metric]))
    return np.asarray(sorted(rows), dtype=float)


@pytest.fixture(scope="module")
def sources() -> dict[str, object]:
    with np.load(ROOT / "paper/figdata/selection_dr_figdata.npz", allow_pickle=False) as archive:
        selection_windows = {key: np.asarray(archive[key]) for key in archive.files}
    return {
        "summary": _json("paper/results_snapshot/summary.json"),
        "demography": _json("paper/results_snapshot/demography_matched.json"),
        "selection": _json("paper/figdata/selection_dr.json"),
        "selection_windows": selection_windows,
    }


@pytest.fixture(scope="module")
def rendered(sources: dict[str, object]):
    module = _load_script("fig_manuscript.py", "figure2_contract_renderer")
    captured: dict[str, object] = {}
    module.save = lambda figure, stem: captured.update(figure=figure, stem=stem)
    module.style()
    module.figure1(
        sources["summary"],
        sources["selection"],
        sources["selection_windows"],
        sources["demography"],
    )
    figure = captured["figure"]
    yield module, figure, captured["stem"]
    plt.close(figure)


def test_figure2_is_bound_to_one_reproducible_output_and_producer() -> None:
    assert (
        r"\includegraphics[width=\textwidth]{figures/fig1_method_validation.pdf}"
        in MAIN
    )
    assert MAIN.count(r"\label{fig:qualification}") == 1
    manifest = _json("paper/figure_provenance.json")
    records = [
        row
        for row in manifest["figures"]
        if row["output"] == "paper/manuscript/figures/fig1_method_validation.pdf"
    ]
    assert len(records) == 1
    record = records[0]
    assert record["producer"] == "scripts/fig_manuscript.py"
    assert "scripts/fig_manuscript.py" in record["command"]
    assert "fig1" in record["command"]
    inputs = {row["path"] for row in record["inputs"]}
    assert {
        "paper/results_snapshot/summary.json",
        "paper/results_snapshot/demography_matched.json",
        "paper/figdata/selection_dr.json",
        "paper/figdata/selection_dr_figdata.npz",
    } <= inputs


def test_figure2_panel_layout_titles_axes_and_letters(rendered) -> None:
    _module, figure, stem = rendered
    assert stem == "fig1_method_validation"
    assert len(figure.axes) == 6
    assert [axis.get_title(loc="left") for axis in figure.axes] == [
        "Reconstruction of map shape at 25 kb",
        "Calibration of intervals",
        "Calibration of rates",
        "Cost-accuracy trade-off",
        "SLiM: map shape",
        "SLiM: rate scale",
    ]
    letters = [[text.get_text() for text in axis.texts if text.get_text() in "abcde"] for axis in figure.axes]
    assert letters == [["a"], ["b"], ["c"], ["d"], ["e"], []]
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "Pearson $r$ at 25 kb",
        "Empirical coverage",
        "median estimated / true rate",
        "Pearson $r$ at 25 kb",
        "Pearson $r$ at 25 kb",
        "median estimated / true",
    ]


@pytest.mark.parametrize("method", METHODS)
def test_panel_a_every_primary_point_matches_its_canonical_record(
    rendered, sources: dict[str, object], method: str
) -> None:
    module, figure, _stem = rendered
    axis = figure.axes[0]
    line = next(line for line in axis.lines if line.get_label() == DISPLAY_NAMES[method])
    np.testing.assert_array_equal(line.get_xdata(), np.arange(len(SCENARIOS)))
    np.testing.assert_allclose(
        line.get_ydata(),
        _expected(sources["summary"], sources["demography"], method, "pearson"),
        rtol=0,
        atol=1e-15,
    )
    assert line.get_marker() == METHOD_MARKERS[method]
    expected_color = {
        "fastrho": module.BLUE,
        "pyrho": module.GREEN,
        "relernn": module.GRAY,
    }[method]
    assert to_rgba(line.get_color()) == to_rgba(expected_color)


def test_panel_a_open_symbols_and_connectors_encode_constant_history(
    rendered, sources: dict[str, object]
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[0]
    open_points = np.asarray(
        sorted(tuple(collection.get_offsets()[0]) for collection in axis.collections),
        dtype=float,
    )
    np.testing.assert_allclose(
        open_points,
        _open_expected(sources["demography"], "pearson", panel="shape"),
        rtol=0,
        atol=1e-15,
    )
    assert len(axis.collections) == 4
    for collection in axis.collections:
        np.testing.assert_allclose(collection.get_facecolors()[0], (1, 1, 1, 1))
    connectors = [line for line in axis.lines if line.get_linestyle() == ":"]
    assert len(connectors) == 4
    for connector in connectors:
        assert connector.get_xdata()[0] < connector.get_xdata()[1]
    assert not any("bottleneck/expansion competitors" in text.get_text() for text in axis.texts)
    legend = axis.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "constant",
        "matched",
    ]
    assert to_rgba(legend.legend_handles[0].get_markerfacecolor()) == to_rgba("white")
    assert to_rgba(legend.legend_handles[1].get_markerfacecolor()) == to_rgba("black")


def test_panel_a_direct_labels_are_complete_and_unambiguous(rendered) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[0]
    direct_labels = {text.get_text() for text in axis.texts} & set(DISPLAY_NAMES.values())
    assert direct_labels == set(DISPLAY_NAMES.values())
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [row[0] for row in SCENARIOS]


def test_panel_b_curve_is_exact_and_summary_statistics_are_caption_only(
    rendered, sources: dict[str, object]
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[1]
    heldout = sources["summary"]["heldout"]
    curve = axis.lines[1]
    np.testing.assert_allclose(curve.get_xdata(), heldout["coverage_curve"]["nominal"])
    np.testing.assert_allclose(curve.get_ydata(), heldout["coverage_curve"]["empirical"])
    np.testing.assert_allclose(axis.lines[0].get_xdata(), (0.45, 1.0))
    np.testing.assert_allclose(axis.lines[0].get_ydata(), (0.45, 1.0))
    visible = {text.get_text() for text in axis.texts}
    assert not any("intervals" in text for text in visible)
    assert not any("nominal 95%" in text for text in visible)
    assert not any("$N_e$" in text for text in visible)


@pytest.mark.parametrize("method", METHODS)
def test_panel_c_every_bias_point_matches_its_canonical_record(
    rendered, sources: dict[str, object], method: str
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[2]
    collection = next(
        collection
        for collection in axis.collections
        if collection.get_label() == DISPLAY_NAMES[method]
    )
    offsets = np.asarray(collection.get_offsets(), dtype=float)
    x_offset = {"fastrho": -0.22, "pyrho": 0.0, "relernn": 0.22}[method]
    np.testing.assert_allclose(offsets[:, 0], np.arange(len(SCENARIOS)) + x_offset)
    np.testing.assert_allclose(
        offsets[:, 1],
        _expected(sources["summary"], sources["demography"], method, "bias_ratio"),
        rtol=0,
        atol=1e-15,
    )


def test_panel_c_scale_reference_open_symbols_and_legend_are_exact(
    rendered, sources: dict[str, object]
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[2]
    assert axis.get_yscale() == "log"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [row[0] for row in SCENARIOS]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        "fastrho",
        "pyrho",
        "ReLERNN",
    ]
    open_collections = [
        collection
        for collection in axis.collections
        if len(collection.get_offsets()) == 1
    ]
    open_points = np.asarray(
        sorted(tuple(collection.get_offsets()[0]) for collection in open_collections),
        dtype=float,
    )
    np.testing.assert_allclose(
        open_points,
        _open_expected(sources["demography"], "bias_ratio", panel="scale"),
        rtol=0,
        atol=1e-15,
    )
    for collection in open_collections:
        np.testing.assert_allclose(collection.get_facecolors()[0], (1, 1, 1, 1))
    reference = [line for line in axis.lines if np.allclose(line.get_ydata(), 1)]
    assert len(reference) == 1
    assert len([line for line in axis.lines if line.get_linestyle() == ":"]) == 4


def test_panel_d_cost_accuracy_points_labels_and_log_scale_are_exact(
    rendered, sources: dict[str, object]
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[3]
    assert axis.get_xscale() == "log"
    assert axis.get_xlabel() == "relative wall-clock cost per dataset"
    observed = np.asarray(
        [collection.get_offsets()[0] for collection in axis.collections], dtype=float
    )
    expected = np.asarray(
        [
            (
                sources["summary"]["timings"][method],
                sources["summary"]["const_n20"]["scales"]["25kb"][method]["pearson"],
            )
            for method in METHODS
        ]
    )
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)
    annotations = {text.get_text(): text for text in axis.texts if hasattr(text, "xy")}
    assert set(annotations) == set(DISPLAY_NAMES.values())
    for method, display in DISPLAY_NAMES.items():
        np.testing.assert_allclose(annotations[display].xy, expected[METHODS.index(method)])


def test_panel_d_rejects_an_unrecorded_runtime_instead_of_using_a_fallback(sources) -> None:
    module = _load_script("fig_manuscript.py", "figure2_missing_runtime_renderer")
    changed = copy.deepcopy(sources["summary"])
    changed.pop("timings")
    module.save = lambda _figure, _stem: None
    before = set(plt.get_fignums())
    with pytest.raises(KeyError, match="timings"):
        module.figure1(
            changed,
            sources["selection"],
            sources["selection_windows"],
            sources["demography"],
        )
    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def test_panel_e_shape_points_and_selected_conditions_are_exact(
    rendered, sources: dict[str, object]
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[4]
    records = {record["name"]: record for record in sources["selection"]["conditions"]}
    selected = [records[name] for name in ("neutral", "bgsint_4", "compl_7")]
    assert [(record["n_regions"], record["exon_frac"]) for record in selected[:2]] == [
        (40, 0.25),
        (40, 0.22),
    ]
    assert selected[2]["n_regions"] == 40
    assert selected[2]["sweep_s"] == 0.05
    assert selected[2]["sweep_target"] == 1.0
    assert selected[2]["soft_k"] == 1
    lines = {line.get_label(): line for line in axis.lines}
    np.testing.assert_allclose(
        lines["fastrho"].get_ydata(), [record["fastrho_cmn_25kb"][0] for record in selected]
    )
    np.testing.assert_allclose(
        lines["pyrho"].get_ydata(), [record["pyrho_25kb"][0] for record in selected]
    )
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        "neutral",
        "BGS",
        "hard\nsweep",
    ]


def test_panel_e_scale_is_rederived_from_the_committed_window_arrays(
    rendered, sources: dict[str, object]
) -> None:
    _module, figure, _stem = rendered
    axis = figure.axes[5]
    expected: dict[str, list[float]] = {"fastrho": [], "pyrho": []}
    for method in expected:
        for condition in ("neutral", "sweep"):
            truth = sources["selection_windows"][f"calib_true_{condition}"]
            predicted = sources["selection_windows"][f"calib_{method}_{condition}"]
            keep = (
                np.isfinite(truth)
                & np.isfinite(predicted)
                & (truth > 0)
                & (predicted > 0)
            )
            assert keep.sum() > 0
            expected[method].append(float(np.median(predicted[keep] / truth[keep])))
    method_lines = axis.lines[:2]
    np.testing.assert_allclose(method_lines[0].get_ydata(), expected["fastrho"])
    np.testing.assert_allclose(method_lines[1].get_ydata(), expected["pyrho"])
    assert [line.get_marker() for line in method_lines] == ["o", "s"]
    assert axis.get_ylim() == pytest.approx((0.45, 1.03))
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["neutral", "hard\nsweep"]
    assert not any(key.startswith("calib_") and "bgs" in key for key in sources["selection_windows"])
    reference = [line for line in axis.lines if np.allclose(line.get_ydata(), 1)]
    assert len(reference) == 1
    assert "1 = correct scale" in {text.get_text() for text in axis.texts}


def test_panel_e_legend_names_only_the_methods_that_were_run(rendered) -> None:
    _module, figure, _stem = rendered
    shape_axis, scale_axis = figure.axes[4:]
    assert [text.get_text() for text in shape_axis.get_legend().get_texts()] == [
        "fastrho",
        "pyrho",
    ]
    visible_text = " ".join(
        text.get_text()
        for axis in (shape_axis, scale_axis)
        for text in axis.texts + list(axis.get_xticklabels()) + list(axis.get_yticklabels())
    )
    assert "ReLERNN" not in visible_text
    assert r"\relernn\rev{ was omitted from panel e" in _caption()


def test_no_unaccounted_data_artists_can_be_added_to_figure2(rendered) -> None:
    """Force new plotted artists to receive an explicit source-aware test."""

    _module, figure, _stem = rendered
    assert [(len(axis.lines), len(axis.collections)) for axis in figure.axes] == [
        (7, 4),
        (2, 0),
        (5, 7),
        (0, 3),
        (2, 0),
        (3, 0),
    ]


def test_figure2_caption_parameters_match_the_executable_simulation_design(sources) -> None:
    bench = _load_script("bench.py", "figure2_benchmark_design")
    from fastrho.simulate import RecombPriors

    defaults = bench.Config("contract")
    priors = RecombPriors()
    assert defaults.seq_len == 2_000_000
    assert 2 * defaults.n_dip == 20
    assert defaults.Ne == 10_000
    assert defaults.mu == 1.5e-8
    assert bench.GRID == 25_000
    assert bench.SYNTHETIC_MEDIAN_RATE == 1e-8
    assert priors.rate_clip == (1e-10, 2e-7)
    for _label, key in SCENARIOS:
        record = sources["summary"][key]
        assert record["n_hap"] == 20
        assert record["Ne"] == (13_000 if key == "real_dog" else 10_000)
    dog_command = (ROOT / "scripts/supplementary.sh").read_text(encoding="utf-8")
    assert "--Ne 13000 --mu 4e-9 --n-dip 10" in dog_command
    slim = _literal_assignments(ROOT / "scripts/slim_gen.py")
    assert slim["L"] == 2_000_000
    assert slim["NE"] == 10_000.0
    assert slim["MU"] == 1.5e-8

    design_text = MAIN + SI
    for phrase in (
        "Independent 40-region SLiM experiments",
        "20 sampled haplotypes",
        r"$N_e=10{,}000$",
        r"mutation rate }$1.5\times10^{-8}$",
        r"background selection affecting 22\% exonic sequence",
        r"hard sweep with }$s=0.05$",
    ):
        assert phrase in design_text


def test_figure2_caption_panel_descriptions_match_the_rendered_encodings(rendered) -> None:
    _module, figure, _stem = rendered
    caption = _caption()
    assert "(a) Pearson correlation with the true map at 25 kb" in caption
    assert "(b) Empirical versus nominal interval coverage across 365,280 held-out intervals" in caption
    assert "(c) Median estimated-to-true rate ratio" in caption
    assert "(d) Pearson correlation versus relative wall-clock cost" in caption
    assert "(e) Independent 40-region SLiM experiments" in caption
    assert r"\relernn\rev{ was omitted from panel e" in caption
    assert figure.axes[2].get_yscale() == "log"
    assert figure.axes[3].get_xscale() == "log"


def test_figure2_surrounding_matched_history_numbers_are_exact(sources) -> None:
    block = MAIN.split(r"\rev{In the bottleneck scenario", 1)[1].split(
        r"\rev{Across 365,280 held-out SNP intervals", 1
    )[0]
    demography = sources["demography"]["scenarios"]
    for method, display in (("relernn", r"\relernn"), ("pyrho", r"\pyrho")):
        pairs = []
        for scenario in ("bottleneck", "expansion"):
            arms = demography[scenario][method]["arms"]
            pairs.append((arms["constant"]["25kb"]["pearson"], arms["matched"]["25kb"]["pearson"]))
        assert display in block
        assert f"from {pairs[0][0]:.3f} to {pairs[0][1]:.3f}" in block
        assert f"from {pairs[1][0]:.3f} to {pairs[1][1]:.3f}" in block


def test_figure2_surrounding_coverage_and_runtime_numbers_are_exact(sources) -> None:
    coverage_start = r"\rev{Across 365,280 held-out SNP intervals"
    assert coverage_start in MAIN
    block = MAIN.split(coverage_start, 1)[1].split(
        r"\begin{figure*}", 1
    )[0]
    heldout = sources["summary"]["heldout"]
    index = heldout["coverage_curve"]["nominal"].index(0.95)
    empirical = heldout["coverage_curve"]["empirical"][index]
    assert f"{heldout['n_intervals']:,} held-out SNP intervals" in coverage_start
    assert f"{100 * empirical:.1f}\\% of the time" in block
    timings = sources["summary"]["timings"]
    expected_costs = (
        f"were {timings['fastrho']:.0f}, {timings['pyrho']:.0f}, "
        f"and {timings['relernn']:,.0f}"
    )
    assert expected_costs in block
