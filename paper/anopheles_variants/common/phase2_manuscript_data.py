#!/usr/bin/env python3
"""Render checksum-bound Phase 2 results into LaTeX macros and cohort tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pvalue(value: float) -> str:
    if value < 0.001:
        return f"{value:.2g}"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inversion_path = args.results / "phase2_2la.json"
    resistance_path = args.results / "phase2_resistance.json"
    pyrho_path = args.results / "phase2_pyrho.json"
    pedigree_path = args.results / "pedigree/phase2_pedigree.json"
    inversion = json.loads(inversion_path.read_text())
    resistance = json.loads(resistance_path.read_text())
    pyrho = json.loads(pyrho_path.read_text())
    pedigree = json.loads(pedigree_path.read_text())
    with args.selection.open(newline="", encoding="utf-8") as handle:
        cohorts = list(csv.DictReader(handle, delimiter="\t"))
    primary = resistance["panels"]["hancock_mechanisms"]["summary"]
    core = resistance["panels"]["core_six"]["summary"]
    surveillance = resistance["panels"]["surveillance_markers"]["summary"]
    deduplicated = resistance["derived_sensitivities"]["overlap_deduplicated_14_region"]
    species = primary["by_species"]
    gambiae = species["Anopheles gambiae"]
    coluzzii = species["Anopheles coluzzii"]
    macros = {
        "PhaseTwoCohortCount": str(len(cohorts)),
        "PhaseTwoDiploidCount": str(sum(int(row["n_diploid"]) for row in cohorts)),
        "PhaseTwoGambiaeCount": str(sum(row["species"] == "Anopheles gambiae" for row in cohorts)),
        "PhaseTwoColuzziiCount": str(sum(row["species"] == "Anopheles coluzzii" for row in cohorts)),
        "PhaseTwoTagSnpCount": str(inversion["tag_snps_matched"]),
        "PhaseTwoPearsonR": number(inversion["pearson_Hexp_depth"][0]),
        "PhaseTwoPearsonP": pvalue(inversion["pearson_Hexp_depth"][1]),
        "PhaseTwoSpearmanR": number(inversion["spearman_Hexp_depth"][0]),
        "PhaseTwoSpearmanP": pvalue(inversion["spearman_Hexp_depth"][1]),
        "PhaseTwoCrossCount": str(pedigree["n_crosses"]),
        "PhaseTwoProgenyCount": str(pedigree["n_progeny"]),
        "PhaseTwoTransmissionCount": str(pedigree["n_parental_transmissions_per_arm"]),
        "PhaseTwoCrossoverCount": str(pedigree["n_events_width_le_1mb"]),
        "PhaseTwoPedigreeWindowCount": str(pedigree["n_supported_5mb_windows"]),
        "PhaseTwoPedigreeR": number(pedigree["spearman_5mb"][0]),
        "PhaseTwoPedigreeP": pvalue(pedigree["spearman_5mb"][1]),
        "PhaseTwoResistanceRatio": number(primary["median_ratio"]),
        "PhaseTwoCoreResistanceRatio": number(core["median_ratio"]),
        "PhaseTwoSurveillanceResistanceRatio": number(surveillance["median_ratio"]),
        "PhaseTwoDeduplicatedResistanceRatio": number(deduplicated["median_ratio"]),
        "PhaseTwoResistanceSignificantCount": str(primary["n_nominal_p_lt_0_05"]),
        "PhaseTwoResistancePopulationCount": str(primary["n_cohorts"]),
        "PhaseTwoGambiaeResistanceRatio": number(gambiae["median_ratio"]),
        "PhaseTwoColuzziiResistanceRatio": number(coluzzii["median_ratio"]),
        "PhaseTwoPyrhoMeanR": number(pyrho["mean_spearman_matched"]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    macro_path = args.out / "phase2_numbers.tex"
    macro_path.write_text("% Generated from checksum-bound Phase 2 results.\n" + "\n".join(
        f"\\global\\def\\{name}{{{value}}}" for name, value in macros.items()
    ) + "\n")
    table_path = args.out / "phase2_cohorts.tex"
    lines = [
        r"\begin{tabular}{llllr}",
        r"\toprule",
        r"Cohort & Release population & Species & Country & Diploids \\",
        r"\midrule",
    ]
    for row in cohorts:
        species_name = r"\textit{An. " + row["species"].split()[-1] + "}"
        country = row["country"].replace("_", r"\_")
        lines.append(f"{row['cohort'].replace('_', r'\_')} & {row['release_population']} & {species_name} & {country} & {row['n_diploid']} " + r"\\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    table_path.write_text("\n".join(lines) + "\n")
    provenance = {
        "schema_version": 1,
        "inputs": {str(path): sha256(path) for path in (
            inversion_path, resistance_path, pyrho_path, pedigree_path, args.selection
        )},
        "outputs": {str(path): sha256(path) for path in (macro_path, table_path)},
        "macros": macros,
    }
    (args.out / "phase2_manuscript_data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(macro_path)


if __name__ == "__main__":
    main()
