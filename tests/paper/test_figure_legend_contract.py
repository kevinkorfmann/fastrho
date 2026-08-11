"""Reader-facing contracts for every active figure legend.

The detailed Figure 2 contract verifies every plotted number.  This companion
test guards the information a reader needs to interpret all 14 active figures:
panel coverage, visual encodings, denominators, uncertainty, scale, and the two
live legends whose omissions prompted this audit.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from manuscript_source import MAIN, SI
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[2]


def _load_script(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _caption(label: str) -> str:
    manuscript = MAIN if f"\\label{{{label}}}" in MAIN else SI
    label_start = manuscript.index(f"\\label{{{label}}}")
    block_start = manuscript.rfind(r"\begin{figure", 0, label_start)
    block_end = manuscript.index(r"\end{figure", label_start)
    block = manuscript[block_start:block_end]
    match = re.search(r"\\caption\{(.*)\}\s*\\label", block, flags=re.DOTALL)
    assert match is not None, label
    return match.group(1)


# Every active figure is listed here, so adding a new manuscript figure requires
# an explicit decision about its reader-facing legend.
EXPECTED_PANELS = {
    "fig:method": "ab",
    "fig:qualification": "abcde",
    "fig:history": "abcdefg",
    "fig:anopheles": "abcdef",
    "fig:phase2-pedigree": "abc",
    "fig:redpoll": "abc",
    "fig:treeoflife": "",
    "fig:demography-mating-limits": "ab",
    "fig:arabis-cross": "abcdef",
    "fig:resolution": "abc",
    "fig:gene-conversion": "abc",
    "fig:unphased": "ab",
    "fig:relernn-scale": "ab",
    "fig:phase2-pyrho": "",
}


# These are semantic requirements, not prose snapshots.  They deliberately
# allow rewrites while preventing removal of the scale, denominator, uncertainty,
# or mark/color definitions needed to read a figure without searching Methods.
REQUIRED_PHRASES = {
    "fig:method": ("17-value input vector", "six encoder", "four decoder", "Gaussian distribution", "$N_e$"),
    "fig:qualification": ("25-kb resolution", "365,280 held-out intervals", "simulated $N_e$", "estimated-to-true rate", "wall-clock cost", "hard sweep"),
    "fig:history": ("Campbell dog pedigree map", "33 wolves", "120 shared maps", "diamond", "all five", "Salom", "Rowan", "clean-simulation recovery"),
    "fig:anopheles": ("Phase 2", "nine populations", "five chromosome arms", "Thin lines", "thick lines", "15 selected resistance regions", "100-kb rates", "4,000 population-bootstrap samples"),
    "fig:phase2-pedigree": ("all 11 crosses", "nine wild populations", "5-Mb windows", "32 supported", "Marker shapes", "12,869", "circular shifts", "broad resolution"),
    "fig:redpoll": ("37 A/A homokaryotypes", "seven inferred A/B heterokaryotypes", "28 B/B homokaryotypes", "500-kb", "2-Mb", "18.9--75.0-Mb", "matched to the homokaryotype sample sizes"),
    "fig:treeoflife": ("representative chromosome", "cobalt curves", "unphased genotypes", "dotted gray", "normalized within species", "100 kb", "independent samples"),
    "fig:demography-mating-limits": ("120 paired", "100-kb", "lines are medians", "vertical connectors", "Gray estimates", "cobalt estimates", "Using simulations", "Dashed black", "dotted cobalt", "two map references"),
    "fig:arabis-cross": ("742 offspring", "12", "25", "2-Mb", "open symbols", "95\\% chromosome-bootstrap", "structured-selfing"),
    "fig:resolution": ("2 Mb to 25 kb", "exact maps", "all three methods", "same dashed HapMap"),
    "fig:gene-conversion": ("25-kb", "four", "three mean tract lengths", "95\\%", "2,000 bootstrap", "24 recombination-map replicates", "divided by its mean"),
    "fig:unphased": ("phased", "unphased", "Points display species", "Dashed reference maps", "cobalt predictions", "95\\% prediction limits"),
    "fig:relernn-scale": ("hotspot length", "per-SNP", "standard errors", "25-kb", "GRU"),
    "fig:phase2-pyrho": ("Solid cobalt", "dashed gray", "20-haplotype", "3R", "6--14 Mb", "100-kb", "cohort's median", "Spearman"),
}


@pytest.fixture(scope="module")
def monkeypatch_module():
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


def test_every_active_figure_has_one_complete_caption_contract() -> None:
    labels = re.findall(r"\\label\{(fig:[^}]+)\}", MAIN + SI)
    assert set(labels) == set(EXPECTED_PANELS)
    assert len(labels) == len(set(labels)) == len(EXPECTED_PANELS)
    assert set(REQUIRED_PHRASES) == set(EXPECTED_PANELS)

    for label, panels in EXPECTED_PANELS.items():
        caption = _caption(label)
        assert len(caption.split()) >= 15, label
        for panel in panels:
            assert f"({panel})" in caption, (label, panel)
        for phrase in REQUIRED_PHRASES[label]:
            assert phrase in caption, (label, phrase)


@pytest.mark.parametrize("forbidden", ("paper/", "scripts/", "figdata", ".json", ".yaml"))
def test_figure_captions_never_expose_internal_file_paths(forbidden: str) -> None:
    for label in EXPECTED_PANELS:
        assert forbidden not in _caption(label), (label, forbidden)


@pytest.fixture(scope="module")
def treeoflife_figure(monkeypatch_module):
    module = _load_script("fig_treeoflife_panel.py", "legend_contract_treeoflife")
    captured = {}
    monkeypatch_module.setattr(Figure, "savefig", lambda *_args, **_kwargs: None)
    monkeypatch_module.setattr(
        module.ps,
        "save",
        lambda figure, *_args, **_kwargs: captured.update(figure=figure),
    )
    module.main()
    yield module, captured["figure"]
    plt.close(captured["figure"])


def test_treeoflife_live_legend_and_method_color_match_the_caption(treeoflife_figure) -> None:
    module, figure = treeoflife_figure
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        "fastrho",
        "published map",
        "pyrho",
    ]
    assert to_rgba(legend.legend_handles[2].get_color()) == to_rgba(module.ps.C["pyrho"])
    pyrho_lines = [
        line
        for line in figure.axes[0].lines
        if to_rgba(line.get_color()) == to_rgba(module.ps.C["pyrho"])
        and line.get_linewidth() == pytest.approx(0.65)
        and line.get_linestyle() == "--"
    ]
    assert pyrho_lines
    metric_text = [
        text
        for text in figure.axes[0].texts
        if text.get_text() in {"vs map", "canid model", "selfing model", "split repeat."}
        or text.get_text().startswith("vs pyrho")
    ]
    assert len({text.get_position() for text in metric_text}) == len(metric_text)
