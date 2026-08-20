"""The published documentation stays small, executable, and honest about artifacts."""

from __future__ import annotations

import ast
import json
import re

import paperlib as P
import pytest

pytestmark = pytest.mark.consistency

DOCS = P.REPO_ROOT / "docs"
PAGES = {
    "index.md",
    "quickstart.md",
    "python-api.md",
    "simulation.md",
    "your-data.md",
    "interpretation.md",
    "data.md",
    "checkpoints.md",
}


def _public_text() -> str:
    return "\n".join((DOCS / page).read_text(encoding="utf-8") for page in sorted(PAGES))


def test_public_docs_are_intentionally_minimal() -> None:
    actual = {path.name for path in DOCS.glob("*.md")}
    assert actual == PAGES
    assert not [path for path in DOCS.rglob("*.ipynb") if "_build" not in path.parts]
    assert not (DOCS / "models").exists()
    assert not (DOCS / "reproducibility").exists()


def test_user_api_surface_is_documented() -> None:
    api = (DOCS / "python-api.md").read_text(encoding="utf-8")
    expected = {
        "quick_map_from_vcf",
        "load_model",
        "predict_map_from_vcf",
        "predict_map_from_ts",
        "predict_map_from_genotype_matrix",
        "read_vcf",
        "vcf_contigs",
        "to_dataframe",
        "write_bed",
        "rebin_to_windows",
    }
    assert not sorted(name for name in expected if name not in api)


def test_vcf_sample_contract_is_prominent_and_explicit() -> None:
    quickstart = (DOCS / "quickstart.md").read_text(encoding="utf-8")
    guide = (DOCS / "your-data.md").read_text(encoding="utf-8")
    api = (DOCS / "python-api.md").read_text(encoding="utf-8")
    for text in (quickstart, guide, api):
        assert "every sample" in text.lower() or "all vcf sample" in text.lower()
        assert "1,000" in text
        assert "10--100" in text
    assert "will not subsample" in guide
    assert "gm.shape[0] // 2" in guide


def test_msprime_example_reaches_the_public_tree_sequence_api() -> None:
    api = (DOCS / "simulation.md").read_text(encoding="utf-8")
    expected = {
        "msprime.RateMap",
        "msprime.sim_ancestry",
        "msprime.sim_mutations",
        "msprime.BinaryMutationModel",
        "fastrho.predict_map_from_ts",
        "fastrho.rebin_to_windows",
        "fastrho.write_bed",
        "get_cumulative_mass",
        "np.corrcoef",
        "scale_ratio",
        "log10_rmse",
    }
    assert not sorted(name for name in expected if name not in api)


def test_model_guide_covers_every_registered_model_and_view() -> None:
    guide = (DOCS / "your-data.md").read_text(encoding="utf-8")
    registry = json.loads((P.REPO_ROOT / "fastrho" / "model_registry.json").read_text())
    for model in registry["models"]:
        assert model["id"] in guide
        for input_mode in model["supported_inputs"]:
            assert f"`{input_mode}`" in guide


def test_checkpoint_page_covers_user_and_paper_support_registries() -> None:
    page = (DOCS / "checkpoints.md").read_text(encoding="utf-8")
    registry = json.loads((P.REPO_ROOT / "fastrho" / "model_registry.json").read_text())
    support = json.loads((P.REPO_ROOT / "reproduce" / "checkpoints.json").read_text())
    for model in registry["models"]:
        assert f"`{model['id']}`" in page
    for group in support["groups"]:
        assert f"`{group['id']}`" in page


def test_hero_uses_the_static_paper_schematic() -> None:
    schematic = DOCS / "_static_public" / "fastrho_schematic.png"
    evaluation = DOCS / "_static" / "msprime_evaluation.png"
    index = (DOCS / "index.md").read_text(encoding="utf-8")
    assert schematic.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "_static_public/fastrho_schematic.png" in index
    assert "anim_inference.gif" not in index
    assert evaluation.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert evaluation.stat().st_size < 1_000_000
    assert "generative map" not in _public_text().lower()
    assert "generative profile" not in _public_text().lower()
    assert "one forward pass" not in _public_text().lower()


@pytest.mark.parametrize("page", sorted(PAGES))
def test_python_examples_parse(page: str) -> None:
    text = (DOCS / page).read_text(encoding="utf-8")
    for index, source in enumerate(re.findall(r"```python\n(.*?)```", text, re.DOTALL), start=1):
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - assertion gives a clearer page/block label
            raise AssertionError(f"invalid Python in {page} block {index}: {exc}") from exc


def test_paper_artifact_availability_is_visible() -> None:
    registry = json.loads((P.REPO_ROOT / "fastrho" / "model_registry.json").read_text())
    models = registry["models"]
    assert models
    assert all(model["status"] == "available" for model in models)
    assert all(model.get("archive_url") and model.get("archive_sha256") for model in models)
    text = _public_text()
    assert "public, checksummed download" in text
    assert "artifact-pending" not in text
    for model in models:
        assert model["id"] in text
