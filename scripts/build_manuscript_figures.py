#!/usr/bin/env python3
"""Rebuild and checksum every figure included by the manuscript or SI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper" / "figure_provenance.json"
SOURCE_DATE_EPOCH = "1420070400"
FIGURE_PYTHON = ["uv", "run", "--extra", "figures", "python"]


GROUPS: tuple[dict[str, object], ...] = (
    {
        "producer": "paper/manuscript/figures/fastrho_header_mark.svg",
        "command": [],
        "inputs": ["paper/manuscript/figures/fastrho_header_mark.svg"],
        "outputs": ["paper/manuscript/figures/fastrho_header_mark.pdf"],
        "kind": "decorative-static",
    },
    {
        "producer": "scripts/fig_manuscript.py",
        "command": [
            *FIGURE_PYTHON,
            "scripts/fig_manuscript.py",
            "fig1",
        ],
        "inputs": [
            "paper/results_snapshot/summary.json",
            "paper/results_snapshot/demography_matched.json",
            "paper/figdata/selection_dr.json",
            "paper/figdata/selection_dr_figdata.npz",
        ],
        "outputs": [
            "paper/manuscript/figures/fig1_method_validation.pdf",
        ],
    },
    {
        "producer": "paper/anopheles_variants/common/plot_phase2.py",
        "command": [
            *FIGURE_PYTHON,
            "paper/anopheles_variants/common/plot_phase2.py",
            "--maps",
            "paper/anopheles_variants/phase2/maps",
            "--selection",
            "paper/anopheles_variants/phase2/cohorts/selection.tsv",
            "--results",
            "paper/anopheles_variants/phase2/results",
            "--out",
            "paper/manuscript/figures",
        ],
        "inputs": [
            "paper/anopheles_variants/common/plot_phase2.py",
            "paper/anopheles_variants/phase2/maps/*.npz",
            "paper/anopheles_variants/phase2/cohorts/selection.tsv",
            "paper/anopheles_variants/phase2/results/phase2_2la.json",
            "paper/anopheles_variants/phase2/results/phase2_resistance.json",
            "paper/anopheles_variants/phase2/results/phase2_pyrho.json",
            "paper/anopheles_variants/phase2/results/pedigree/phase2_pedigree.json",
        ],
        "outputs": [
            "paper/manuscript/figures/fig_phase2_anopheles.pdf",
            "paper/manuscript/figures/fig_phase2_pedigree.pdf",
            "paper/manuscript/figures/fig_phase2_pyrho.pdf",
        ],
    },
    {
        "producer": "paper/manuscript/figures/fig1_complete.tex",
        "command": [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "fig1_architecture.tex",
            "fig1_complete.tex",
        ],
        "cwd": "paper/manuscript/figures",
        "inputs": [
            "paper/manuscript/figures/fig1_architecture.tex",
            "paper/manuscript/figures/fig1_complete.tex",
        ],
        "outputs": ["paper/manuscript/figures/fig1_complete.pdf"],
    },
    {
        "producer": "scripts/fig_manuscript_history.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_manuscript_history.py"],
        "inputs": [
            "paper/figdata/dog_fig.npz",
            "paper/figdata/canid_empirical_scale.json",
            "paper/figdata/canid_empirical_ld.json",
            "paper/figdata/selfer_chroms.npz",
            "paper/figdata/selfer_ceiling.json",
        ],
        "outputs": ["paper/manuscript/figures/fig2_history_rescue.pdf"],
    },
    {
        "producer": "scripts/fig_redpoll_karyotype.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_redpoll_karyotype.py"],
        "inputs": [
            "paper/figdata/fieldguide_redpoll.npz",
            "paper/figdata/redpoll_karyotype_maps.npz",
            "paper/figdata/redpoll_karyotype_maps.json",
            "paper/figdata/redpoll_karyotype_null.json",
            "paper/figdata/redpoll_karyotype_ld.json",
        ],
        "outputs": ["paper/figures/fig_redpoll_karyotype.pdf"],
    },
    {
        "producer": "scripts/fig_treeoflife_panel.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_treeoflife_panel.py"],
        "inputs": ["paper/figdata/transect.json", "paper/figdata/transect_pyrho.json"],
        "optional_inputs": [
            "paper/figdata/silhouettes/human.png",
            "paper/figdata/silhouettes/cattle.png",
            "paper/figdata/silhouettes/sheep.png",
            "paper/figdata/silhouettes/goat.png",
            "paper/figdata/silhouettes/donkey.png",
            "paper/figdata/silhouettes/dmel.png",
            "paper/figdata/silhouettes/jewelwasp.png",
            "paper/figdata/silhouettes/athal.png",
            "paper/figdata/silhouettes/aspen.png",
            "paper/figdata/silhouettes/chestnut.png",
        ],
        "outputs": ["paper/figures/fig_treeoflife_panel.pdf"],
    },
    {
        "producer": "scripts/fig_si_unique.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_si_unique.py"],
        "inputs": [
            "paper/figdata/dog_fig.npz",
            "paper/figdata/selfer_ceiling.json",
            "paper/figdata/identifiability.json",
            "paper/figdata/mechanism.json",
            "paper/figdata/relernn_showdown.npz",
            "paper/figdata/realmaps.npz",
            "paper/figdata/repro_showdown.npz",
            "paper/results_snapshot/demography_matched.json",
            "paper/figdata/selection_dr_figdata.npz",
        ],
        "outputs": [
            "paper/figures/fig_si_demography_mating_limits.pdf",
            "paper/figures/fig_si_resolution.pdf",
            "paper/figures/fig_si_relernn_mechanisms.pdf",
        ],
    },
    {
        "producer": "scripts/fig_arabis_cross.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_arabis_cross.py"],
        "inputs": [
            "paper/figdata/arabis_cross.npz",
            "paper/figdata/arabis_cross_structured.npz",
            "paper/results_snapshot/arabis_cross.json",
            "paper/results_snapshot/arabis_cross_smalln.json",
            "paper/results_snapshot/arabis_cross_structured.json",
            "paper/results_snapshot/arabis_window_diagnostics.json",
        ],
        "outputs": ["paper/figures/fig_si_arabis_cross.pdf"],
    },
    {
        "producer": "scripts/fig_gene_conversion.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_gene_conversion.py"],
        "inputs": [
            "paper/results_snapshot/gene_conversion.json",
            "paper/figdata/gene_conversion.npz",
        ],
        "outputs": ["paper/figures/fig_gene_conversion.pdf"],
    },
    {
        "producer": "scripts/fig_unphased.py",
        "command": [*FIGURE_PYTHON, "scripts/fig_unphased.py"],
        "inputs": ["paper/figures/_stdpopsim_*.json"],
        "optional_inputs": ["paper/figures/silhouettes/*.png"],
        "outputs": ["paper/figures/fig5_unphased.pdf"],
    },
)


def _expand(patterns: list[str]) -> tuple[Path, ...]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return tuple(sorted(paths))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest() -> dict[str, object]:
    figures: list[dict[str, object]] = []
    for group in GROUPS:
        required = _expand(list(group["inputs"]))
        for pattern in group["inputs"]:
            if not _expand([str(pattern)]):
                raise FileNotFoundError(f"figure input pattern matched nothing: {pattern}")
        optional = _expand(list(group.get("optional_inputs", [])))
        inputs = required + tuple(path for path in optional if path not in required)
        input_records = [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in inputs
        ]
        for output_name in group["outputs"]:
            output = ROOT / str(output_name)
            if not output.is_file():
                raise FileNotFoundError(f"figure output is missing: {output_name}")
            figures.append(
                {
                    "output": str(output_name),
                    "manuscript_target": f"figures_phase2/{output.name}",
                    "output_sha256": _sha256(output),
                    "producer": group["producer"],
                    "command": list(group["command"]),
                    "cwd": str(group.get("cwd", ".")),
                    "inputs": input_records,
                }
            )
    figures.sort(key=lambda row: str(row["output"]))
    return {
        "schema_version": 1,
        "policy": "Every included PDF has one executable producer and checksummed committed inputs. Optional cosmetic silhouettes are checksummed when present.",
        "deterministic_build_environment": {"SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH},
        "figures": figures,
    }


def run_all() -> None:
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    for group in GROUPS:
        command = [str(part) for part in group["command"]]
        if not command:
            continue
        cwd = ROOT / str(group.get("cwd", "."))
        print(f"[{group['producer']}] {shlex.join(command)}")
        subprocess.run(command, cwd=cwd, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="rebuild all included figures first")
    parser.add_argument("--write-manifest", action="store_true", help="update checksums")
    args = parser.parse_args()
    if args.run:
        run_all()
    manifest = build_manifest()
    if args.write_manifest:
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(MANIFEST)
    else:
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
