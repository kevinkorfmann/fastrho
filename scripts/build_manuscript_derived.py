#!/usr/bin/env python3
"""Recompute compact manuscript summaries from committed result artifacts."""

from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "results_snapshot" / "manuscript_derived.json"
BENCHMARK_TABLE = ROOT / "paper" / "tables" / "main_results.tex"
DEMOGRAPHY_SPECIFICATION_TABLE = (
    ROOT / "paper" / "tables" / "demography_specification.tex"
)
BENCHMARK_SUMMARY = ROOT / "paper" / "results_snapshot" / "summary.json"
PAIRED_DEMOGRAPHY = ROOT / "paper" / "results_snapshot" / "demography_matched.json"
DISPLAYED_SCENARIOS = (
    "const_n20",
    "bottleneck_n20",
    "expansion_n20",
    "real_decode",
    "real_hapmap",
    "real_dog",
)
TABLE_SCENARIOS = (
    "const_n20",
    "const_n40",
    "real_hapmap",
    "real_decode",
    "bottleneck_n20",
    "expansion_n20",
    "real_dog",
    "const_n100",
)
sys.path.insert(0, str(ROOT))


def _dog_diploid_choices() -> tuple[int, ...]:
    """Read the literal simulation choices without importing the msprime script."""

    tree = ast.parse((ROOT / "scripts" / "dog_gen.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "n_dip" for target in node.targets
        ):
            continue
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "int"
            and call.args
        ):
            call = call.args[0]
        if not isinstance(call, ast.Call) or not call.args:
            continue
        try:
            choices = ast.literal_eval(call.args[0])
        except (ValueError, TypeError):
            continue
        return tuple(int(value) for value in choices)
    raise ValueError("could not recover n_dip choices from scripts/dog_gen.py")


def _displayed_benchmark_summary() -> dict[str, object]:
    """Bind the exact six plotted 25-kb values to one compact prose source."""

    summary = json.loads(BENCHMARK_SUMMARY.read_text(encoding="utf-8"))
    paired = (
        json.loads(PAIRED_DEMOGRAPHY.read_text(encoding="utf-8"))
        if PAIRED_DEMOGRAPHY.is_file()
        else None
    )
    values: dict[str, list[float]] = {method: [] for method in ("fastrho", "pyrho", "relernn")}
    records: dict[str, dict[str, float]] = {}
    for scenario in DISPLAYED_SCENARIOS:
        records[scenario] = {}
        for method in values:
            if (
                paired is not None
                and scenario in {"bottleneck_n20", "expansion_n20"}
            ):
                paired_scenario = scenario.removesuffix("_n20")
                if method == "fastrho":
                    record = paired["scenarios"][paired_scenario]["fastrho_reference"]["25kb"]
                else:
                    record = paired["scenarios"][paired_scenario][method]["arms"]["matched"]["25kb"]
            else:
                record = summary[scenario]["scales"]["25kb"][method]
            value = float(record["pearson"])
            if not np.isfinite(value):
                raise ValueError(f"nonfinite displayed Pearson value: {scenario}/{method}")
            records[scenario][method] = value
            values[method].append(value)

    method_summaries = {
        method: {
            "values": method_values,
            "minimum": min(method_values),
            "maximum": max(method_values),
        }
        for method, method_values in values.items()
    }
    return {
        "scale_bp": 25_000,
        "scenario_order": list(DISPLAYED_SCENARIOS),
        "competitor_history_for_bottleneck_expansion": (
            "matched" if paired is not None else "constant"
        ),
        "pearson_by_scenario": records,
        "pearson_by_method": method_summaries,
    }


def _benchmark_record(
    summary: dict, paired: dict | None, scenario: str, method: str, scale: str
) -> dict:
    """Return the arm shown in the manuscript table for one method and scenario."""

    if (
        paired is not None
        and scenario in {"bottleneck_n20", "expansion_n20"}
    ):
        paired_scenario = paired["scenarios"][scenario.removesuffix("_n20")]
        if method == "fastrho":
            return paired_scenario["fastrho_reference"][scale]
        return paired_scenario[method]["arms"]["matched"][scale]
    return summary[scenario]["scales"][scale][method]


def build_benchmark_table() -> str:
    """Render the benchmark table from the same constant/matched arms used in Fig. 2."""

    summary = json.loads(BENCHMARK_SUMMARY.read_text(encoding="utf-8"))
    paired = (
        json.loads(PAIRED_DEMOGRAPHY.read_text(encoding="utf-8"))
        if PAIRED_DEMOGRAPHY.is_file()
        else None
    )
    rows = []
    for scenario in TABLE_SCENARIOS:
        cells = []
        for scale in ("100kb", "25kb"):
            for method in ("fastrho", "pyrho", "relernn"):
                value = float(
                    _benchmark_record(summary, paired, scenario, method, scale)["pearson"]
                )
                if not np.isfinite(value):
                    raise ValueError(f"nonfinite table value: {scenario}/{method}/{scale}")
                cells.append(f"{value:.3f}")
        label = scenario.replace("_", r"\_")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return (
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        "config & \\multicolumn{3}{c}{Pearson @100kb} & "
        "\\multicolumn{3}{c}{Pearson @25kb} \\\\\n"
        " & \\fastrho & \\pyrho & \\relernn & \\fastrho & \\pyrho & \\relernn \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def build_demography_specification_table() -> str:
    """Render the paired demographic-specification comparison at 25 kb."""

    paired = json.loads(PAIRED_DEMOGRAPHY.read_text(encoding="utf-8"))
    rows = []
    for scenario in ("bottleneck", "expansion"):
        fixed = paired["scenarios"][scenario]["fastrho_reference"]["25kb"]
        rows.append(
            f"{scenario.capitalize()} & \\fastrho & fixed broad prior & "
            f"{fixed['pearson']:.3f} & {fixed['spearman']:.3f} & "
            f"{fixed['bias_ratio']:.3f} \\\\"
        )
        for method, display in (("pyrho", r"\pyrho"), ("relernn", r"\relernn")):
            for history in ("constant", "matched"):
                record = paired["scenarios"][scenario][method]["arms"][history]["25kb"]
                rows.append(
                    f" & {display} & {history} & {record['pearson']:.3f} & "
                    f"{record['spearman']:.3f} & {record['bias_ratio']:.3f} \\\\"
                )
        rows.append(r"\addlinespace")
    rows.pop()
    return (
        "\\begin{tabular}{lllccc}\n"
        "\\toprule\n"
        "Scenario & Method & Demographic specification & Pearson $r$ & Spearman $\\rho$ & "
        "Median estimated/true \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n"
    )


def build_summary() -> dict[str, object]:
    metadata = json.loads((ROOT / "paper" / "manuscript_metadata.json").read_text())
    agam_validation = json.loads(
        (ROOT / "paper" / "results_snapshot" / "agam_validation.json").read_text()
    )
    with (ROOT / "atlas" / "anopheles" / "manifest.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        agam_manifest = list(csv.DictReader(handle, delimiter="\t"))
    atlas_total_diploids = sum(int(row["n_hap"]) for row in agam_manifest) // 2
    arrangement_total_diploids = sum(
        int(row["n_samples"]) for row in agam_validation["rows"]
    )

    arabis_n = int(metadata["records"]["arabis_cross"]["n_f2_progeny"])
    with (ROOT / "research" / "arabis" / "sample_assignments.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        arabis_assignments = list(csv.DictReader(handle, delimiter="\t"))
    arabis_species_counts = {
        species: sum(row["species"] == species for row in arabis_assignments)
        for species in ("nemorensis", "sagittata")
    }

    redpoll = json.loads((ROOT / "paper" / "figdata" / "redpoll_karyotype_null.json").read_text())
    redpoll_summary: dict[str, object] = {}
    for size in sorted({int(record["size"]) for record in redpoll["records"]}, reverse=True):
        values = np.asarray(
            [
                record["inside_flank_ratio"]
                for record in redpoll["records"]
                if int(record["size"]) == size
            ],
            dtype=float,
        )
        redpoll_summary[str(size)] = {
            "mean_inside_flank_ratio": float(np.mean(values)),
            "min_inside_flank_ratio": float(np.min(values)),
            "max_inside_flank_ratio": float(np.max(values)),
        }
    from fastrho.preprocess import mean_rate_between

    with np.load(ROOT / "paper" / "figdata" / "fieldguide_redpoll.npz") as fieldguide, np.load(
        ROOT / "paper" / "figdata" / "redpoll_karyotype_maps.npz"
    ) as redpoll_maps:
        pooled_breakpoints = np.r_[
            fieldguide["pos_left"][0], fieldguide["pos_right"]
        ]
        pooled_500kb = mean_rate_between(
            pooled_breakpoints, fieldguide["rho_per_bp"], redpoll_maps["edges"]
        )
        centers_mb = np.asarray(redpoll_maps["centers"], dtype=float) / 1_000_000
        inversion_start_mb = float(redpoll_maps["inv_start"]) / 1_000_000
        inversion_end_mb = float(redpoll_maps["inv_end"]) / 1_000_000
    inversion = (centers_mb >= inversion_start_mb) & (centers_mb < inversion_end_mb)
    redpoll_summary["pooled_inside_flank_ratio"] = float(
        np.nanmedian(pooled_500kb[inversion]) / np.nanmedian(pooled_500kb[~inversion])
    )

    redpoll_ld = json.loads((ROOT / "paper" / "figdata" / "redpoll_karyotype_ld.json").read_text())
    ld_lo, ld_hi = 250_000, 500_000

    def redpoll_inside_ld(group: str) -> float:
        record = next(
            row
            for row in redpoll_ld["groups"][group]["inside"]
            if row["distance_lo"] == ld_lo and row["distance_hi"] == ld_hi
        )
        return float(record["mean_r2_corrected"])

    pooled_ld = redpoll_inside_ld("pooled")
    redpoll_ld_summary = {
        "inversion_start_mb": redpoll_ld["inv_start"] / 1_000_000,
        "inversion_end_mb": redpoll_ld["inv_end"] / 1_000_000,
        "distance_lo_kb": ld_lo / 1_000,
        "distance_hi_kb": ld_hi / 1_000,
        "pooled_inside_r2": pooled_ld,
        "arrangement_A_inside_r2": redpoll_inside_ld("arrangement_A"),
        "arrangement_B_inside_r2": redpoll_inside_ld("arrangement_B"),
    }
    redpoll_ld_summary["pooled_to_arrangement_A_ratio"] = (
        pooled_ld / redpoll_ld_summary["arrangement_A_inside_r2"]
    )
    redpoll_ld_summary["pooled_to_arrangement_B_ratio"] = (
        pooled_ld / redpoll_ld_summary["arrangement_B_inside_r2"]
    )
    redpoll_ld_summary["pooled_to_arrangement_A_ratio_rounded"] = round(
        redpoll_ld_summary["pooled_to_arrangement_A_ratio"]
    )
    redpoll_ld_summary["pooled_to_arrangement_B_ratio_rounded"] = round(
        redpoll_ld_summary["pooled_to_arrangement_B_ratio"]
    )

    with np.load(ROOT / "paper" / "figdata" / "arabis_cross_structured.npz") as arrays:
        cross = np.concatenate([arrays[f"chr{i}_cross"] for i in range(1, 9)])
        predicted = np.concatenate([arrays[f"chr{i}_consensus"] for i in range(1, 9)])
    quantile_edges = np.quantile(cross, np.linspace(0, 1, 6))
    quantile_index = np.searchsorted(quantile_edges[1:-1], cross, side="right")
    medians = [float(np.median(predicted[quantile_index == i])) for i in range(5)]

    from fastrho.simulate import RecombPriors

    prior = RecombPriors()
    mutation_lower = 10.0 ** prior.log10_mu_range[0]
    dog_choices = _dog_diploid_choices()

    with (ROOT / "research" / "anopheles_resistance_panels.tsv").open(
        encoding="utf-8", newline=""
    ) as handle:
        resistance_panel = {
            row["locus"]: row for row in csv.DictReader(handle, delimiter="\t")
        }
    resistance_sensitivity = json.loads(
        (
            ROOT / "paper" / "figdata" / "agam_resistance_panel_sensitivity.json"
        ).read_text()
    )
    local_radius_mb = float(resistance_sensitivity["metadata"]["local_flank_mb"])
    rdl_mb = float(resistance_panel["Rdl"]["mb"])
    cyp4j5_mb = float(resistance_panel["Cyp4j5"]["mb"])
    resistance_overlap_kb = max(
        0.0, (2 * local_radius_mb - abs(rdl_mb - cyp4j5_mb)) * 1_000
    )

    return {
        "schema_version": 1,
        "inputs": {
            "metadata": "paper/manuscript_metadata.json",
            "redpoll_null": "paper/figdata/redpoll_karyotype_null.json",
            "redpoll_ld": "paper/figdata/redpoll_karyotype_ld.json",
            "redpoll_pooled_map": "paper/figdata/fieldguide_redpoll.npz",
            "redpoll_stratified_maps": "paper/figdata/redpoll_karyotype_maps.npz",
            "agam_manifest": "atlas/anopheles/manifest.tsv",
            "agam_validation": "paper/results_snapshot/agam_validation.json",
            "resistance_panel": "research/anopheles_resistance_panels.tsv",
            "resistance_sensitivity": (
                "paper/figdata/agam_resistance_panel_sensitivity.json"
            ),
            "arabis_arrays": "paper/figdata/arabis_cross_structured.npz",
            "arabis_sample_assignments": "research/arabis/sample_assignments.tsv",
            "base_prior": "fastrho/simulate.py:RecombPriors",
            "dog_prior": "scripts/dog_gen.py:n_dip",
            "benchmark_summary": "paper/results_snapshot/summary.json",
            "paired_demography": (
                "paper/results_snapshot/demography_matched.json"
                if PAIRED_DEMOGRAPHY.is_file()
                else None
            ),
        },
        "benchmark_displayed_25kb": _displayed_benchmark_summary(),
        "arabis": {
            "n_f2_progeny": arabis_n,
            "n_population_accessions_total": len(arabis_assignments),
            "n_nemorensis_accessions": arabis_species_counts["nemorensis"],
            "n_sagittata_accessions": arabis_species_counts["sagittata"],
            "lower_hundred_strict_bound": (arabis_n // 100) * 100,
            "quintile_medians": medians,
            "highest_to_lowest_median_ratio": medians[-1] / medians[0],
        },
        "redpoll_null": redpoll_summary,
        "redpoll_ld": redpoll_ld_summary,
        "anopheles": {
            "atlas_total_diploid_mosquitoes": atlas_total_diploids,
            "arrangement_call_total_diploid_mosquitoes": arrangement_total_diploids,
        },
        "resistance": {
            "rdl_cyp4j5_window_overlap_kb": resistance_overlap_kb,
            "rdl_cyp4j5_window_overlap_kb_rounded": round(resistance_overlap_kb),
        },
        "prior_display": {
            "mutation_lower_rate": mutation_lower,
            "mutation_lower_mantissa_1dp": round(mutation_lower / 1e-9, 1),
            "dog_min_haplotypes": 2 * min(dog_choices),
            "dog_max_haplotypes": 2 * max(dog_choices),
        },
    }


def main() -> None:
    summary = build_summary()
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    BENCHMARK_TABLE.write_text(build_benchmark_table(), encoding="utf-8")
    DEMOGRAPHY_SPECIFICATION_TABLE.write_text(
        build_demography_specification_table(), encoding="utf-8"
    )
    print(OUTPUT)
    print(BENCHMARK_TABLE)
    print(DEMOGRAPHY_SPECIFICATION_TABLE)


if __name__ == "__main__":
    main()
