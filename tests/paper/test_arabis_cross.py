"""Independent integrity checks for the Arabis F2-cross benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "paper" / "results_snapshot" / "arabis_cross.json"
ARRAYS = ROOT / "paper" / "figdata" / "arabis_cross.npz"
SMALLN_SNAPSHOT = ROOT / "paper" / "results_snapshot" / "arabis_cross_smalln.json"
SMALLN_ARRAYS = ROOT / "paper" / "figdata" / "arabis_cross_smalln.npz"
SMALLN_MANIFEST = ROOT / "paper" / "results_snapshot" / "arabis_smalln_frozen_ensemble.json"
PANEL_DIAGNOSTICS = ROOT / "paper" / "results_snapshot" / "arabis_panel_diagnostics.json"
MAC2_SENSITIVITY = ROOT / "paper" / "results_snapshot" / "arabis_mac2_sensitivity.json"
RESAMPLE_DIAGNOSTICS = ROOT / "paper" / "results_snapshot" / "arabis_resample_diagnostics.json"
COMPOSITION_DIAGNOSTICS = ROOT / "paper" / "results_snapshot" / "arabis_composition_diagnostics.json"
CONDITIONING_SENSITIVITY = ROOT / "paper" / "results_snapshot" / "arabis_conditioning_sensitivity.json"
STRUCTURED_SNAPSHOT = ROOT / "paper" / "results_snapshot" / "arabis_cross_structured.json"
STRUCTURED_ARRAYS = ROOT / "paper" / "figdata" / "arabis_cross_structured.npz"
STRUCTURED_MANIFEST = (
    ROOT / "paper" / "results_snapshot" / "arabis_structured_frozen_ensemble.json"
)
STRUCTURED_GATE = (
    ROOT / "paper" / "results_snapshot" / "arabis_structured_simulation_gate.json"
)
STRUCTURED_DESIGN = (
    ROOT / "paper" / "results_snapshot" / "arabis_structured_design_audit.json"
)
STRUCTURED_COMPLETION = (
    ROOT / "paper" / "results_snapshot" / "arabis_structured_completion_manifest.json"
)
WINDOW_DIAGNOSTICS = (
    ROOT / "paper" / "results_snapshot" / "arabis_window_diagnostics.json"
)
STRUCTURED_PREREG = ROOT / "research" / "arabis" / "structured_selfing_preregistration.json"
SAMPLE_SHEET = ROOT / "research" / "arabis" / "sample_assignments.tsv"


def load():
    return json.loads(SNAPSHOT.read_text()), np.load(ARRAYS, allow_pickle=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_arabis_input_contract_is_frozen() -> None:
    summary, _ = load()
    assert summary["primary_resolution"] == "2Mb"
    assert summary["cross_map_audit"]["markers"] == 2082
    assert summary["source_commit"] == "10c9092ce9e08c16b0a958d4062932262a8c6bc4"
    assert summary["model_campaign"] == "campaign_self2"
    assert summary["model_training_n_lines"] == [50, 80, 120, 156, 200]
    assert set(summary["variant_pipeline_software"]) == {"bwa-mem2", "samtools", "bcftools"}
    assert summary["sample_assignments"]["sha256"] == sha256(SAMPLE_SHEET)
    assert summary["sample_assignments"]["counts"] == {"nemorensis": 12, "sagittata": 25}
    assert set(summary["cross_map_audit"]["chromosomes"]) == {f"chr{i}" for i in range(1, 9)}
    assert summary["population_map_audit"]["nemorensis"]["n_accessions"] == 12
    assert summary["population_map_audit"]["sagittata"]["n_accessions"] == 25
    assert summary["parent_excluded_population_map_audit"]["nemorensis"]["n_accessions"] == 11
    assert summary["parent_excluded_population_map_audit"]["sagittata"]["n_accessions"] == 24
    assert summary["checkpoint_sha256"] == (
        "be51f3ceb7d8d206c15f54e2301d12b3f64e2a366fdc3472dacae7f310d27799"
    )
    assert summary["stats_sha256"] == (
        "f5467d180ab245576bc80906277b98f31c3bcbb081f503319e627e414e4f6efd"
    )


def test_arabis_sample_assignments_match_published_table() -> None:
    with SAMPLE_SHEET.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 37
    assert len({row["accession"] for row in rows}) == 37
    assert sum(row["species"] == "nemorensis" for row in rows) == 12
    assert sum(row["species"] == "sagittata" for row in rows) == 25
    assert next(row for row in rows if row["accession"] == "10")["species"] == "nemorensis"


def test_arabis_primary_arrays_are_chromosome_normalized() -> None:
    _, arrays = load()
    for prefix in ("", "parent_excluded_"):
        for chrom in (f"chr{i}" for i in range(1, 9)):
            n = len(arrays[f"{prefix}{chrom}_cross"])
            assert len(arrays[f"{prefix}{chrom}_edges"]) == n + 1
            for key in ("cross", "nemorensis", "sagittata", "consensus"):
                values = arrays[f"{prefix}{chrom}_{key}"]
                assert np.all(np.isfinite(values))
                assert np.all(values >= 0)
                assert np.isclose(np.mean(values), 1.0, atol=1e-10)


def test_arabis_primary_correlations_rederive() -> None:
    summary, arrays = load()
    primary = summary["resolutions"]["2Mb"]
    cross = np.concatenate([arrays[f"chr{i}_cross"] for i in range(1, 9)])
    for key in ("nemorensis", "sagittata", "consensus"):
        pred = np.concatenate([arrays[f"chr{i}_{key}"] for i in range(1, 9)])
        stored = primary["maps"][key]
        assert np.isclose(spearmanr(cross, pred).statistic, stored["spearman"], atol=1e-12)
        assert np.isclose(pearsonr(cross, pred).statistic, stored["pearson"], atol=1e-12)
        assert stored["n_windows"] == len(cross)
        assert 0 < stored["circular_shift_p_one_sided"] <= 1


def test_arabis_between_species_agreement_rederives() -> None:
    summary, arrays = load()
    a = np.concatenate([arrays[f"chr{i}_nemorensis"] for i in range(1, 9)])
    b = np.concatenate([arrays[f"chr{i}_sagittata"] for i in range(1, 9)])
    stored = summary["resolutions"]["2Mb"]["between_species"]
    assert np.isclose(spearmanr(a, b).statistic, stored["spearman"], atol=1e-12)
    assert np.isclose(pearsonr(a, b).statistic, stored["pearson"], atol=1e-12)


def test_arabis_parent_exclusion_rederives() -> None:
    summary, arrays = load()
    primary = summary["parent_excluded_2Mb"]
    cross = np.concatenate([arrays[f"parent_excluded_chr{i}_cross"] for i in range(1, 9)])
    for key in ("nemorensis", "sagittata", "consensus"):
        pred = np.concatenate([arrays[f"parent_excluded_chr{i}_{key}"] for i in range(1, 9)])
        assert np.isclose(
            spearmanr(cross, pred).statistic,
            primary["maps"][key]["spearman"],
            atol=1e-12,
        )


def test_smalln_campaign_was_frozen_without_cross_map_selection() -> None:
    summary = json.loads(SMALLN_SNAPSHOT.read_text())
    manifest = json.loads(SMALLN_MANIFEST.read_text())
    assert summary["model_campaign"] == "arabis_smalln_self3"
    assert summary["model_training_n_lines"] == [8, 12, 16, 24, 25, 32, 50]
    assert manifest["arabis_cross_map_used_for_selection"] is False
    assert manifest["selection_data"] == "simulated validation only"
    assert len(manifest["members"]) == 5
    assert len({member["checkpoint_sha256"] for member in manifest["members"]}) == 5
    assert len({member["stats_sha256"] for member in manifest["members"]}) == 1
    assert all(0.81 < member["validation_pearson"] < 0.83 for member in manifest["members"])
    assert summary["checkpoint_sha256"].startswith("ensemble_manifest:")


def test_arabis_panel_diagnostic_exposes_minor_count_discontinuity() -> None:
    diagnostic = json.loads(PANEL_DIAGNOSTICS.read_text())
    nem = diagnostic["filtered_panels"]["nemorensis"]
    sag = diagnostic["filtered_panels"]["sagittata"]
    assert nem["n_accessions"] == 12
    assert sag["n_accessions"] == 25
    assert nem["minor_allele_count_spectrum"]["1"] == 101492
    assert np.isclose(nem["singleton_site_fraction"], 101492 / 268562)
    assert sag["singleton_site_fraction"] == 0


def test_arabis_mac2_filter_is_not_a_posthoc_rescue() -> None:
    result = json.loads(MAC2_SENSITIVITY.read_text())
    summary = result["by_design"]["minimum_minor_allele_count_2"]
    assert np.isclose(summary["spearman_subset_vs_cross"]["median"], 0.027542959892791107)
    assert summary["spearman_subset_vs_full"]["median"] < 0.60


def test_arabis_resamples_were_predeclared_and_complete() -> None:
    result = json.loads(RESAMPLE_DIAGNOSTICS.read_text())
    assert result["design_note"] == "Subsets and frozen models were selected without the F2 map."
    assert len(result["panels"]) == 24
    assert {p["design"] for p in result["panels"]} == {"leave_one_out", "stratified_n12"}
    assert result["by_design"]["stratified_n12"]["spearman_subset_vs_cross"]["range"][0] > 0.5
    assert result["by_design"]["leave_one_out"]["spearman_subset_vs_cross"]["range"][1] < 0.03


def test_arabis_rhine_enrichment_does_not_rescue_nemorensis() -> None:
    result = json.loads(COMPOSITION_DIAGNOSTICS.read_text())
    assert len(result["panels"]) == 3
    summary = result["by_design"]["rhine6_plus_disjoint_allopatric_pair"]
    assert summary["spearman_subset_vs_cross"]["range"][1] < -0.13


def test_arabis_doubled_conditioning_does_not_change_conclusion() -> None:
    result = json.loads(CONDITIONING_SENSITIVITY.read_text())
    maps = result["resolutions"]["2Mb"]["maps"]
    assert np.isclose(maps["nemorensis"]["spearman"], -0.022725345064489318)
    assert np.isclose(maps["sagittata"]["spearman"], 0.519199778344482)


def test_smalln_primary_correlations_rederive() -> None:
    summary = json.loads(SMALLN_SNAPSHOT.read_text())
    arrays = np.load(SMALLN_ARRAYS, allow_pickle=False)
    cross = np.concatenate([arrays[f"chr{i}_cross"] for i in range(1, 9)])
    primary = summary["resolutions"]["2Mb"]
    for key in ("nemorensis", "sagittata", "consensus"):
        pred = np.concatenate([arrays[f"chr{i}_{key}"] for i in range(1, 9)])
        assert np.isclose(
            spearmanr(cross, pred).statistic,
            primary["maps"][key]["spearman"],
            atol=1e-12,
        )
    assert primary["maps"]["sagittata"]["spearman"] > 0.5
    assert primary["maps"]["nemorensis"]["spearman"] < 0.0


def test_structured_campaign_was_frozen_before_cross_map_release() -> None:
    manifest = json.loads(STRUCTURED_MANIFEST.read_text())
    gate = json.loads(STRUCTURED_GATE.read_text())
    prereg = json.loads(STRUCTURED_PREREG.read_text())
    assert manifest["campaign"] == "arabis_structured_selfing_v1"
    assert manifest["selection_data"] == "simulated validation only"
    assert manifest["arabis_cross_map_used_for_selection"] is False
    assert len(manifest["members"]) == 7
    assert len({m["checkpoint_sha256"] for m in manifest["members"]}) == 7
    assert len({m["stats_sha256"] for m in manifest["members"]}) == 1
    assert all(0.86 < m["validation_pearson"] < 0.87 for m in manifest["members"])
    assert gate["passed"] is True and gate["failures"] == []
    assert gate["arabis_cross_map_used"] is False
    assert prereg["arabis_cross_map_used_for_training_or_selection"] is False
    assert "Report the result regardless of its sign" in prereg["decision_rule"]


def test_structured_simulation_gate_passes_every_registered_stratum() -> None:
    gate = json.loads(STRUCTURED_GATE.read_text())
    expected_n = {
        "panmictic": 469,
        "diffuse_island": 519,
        "nemorensis_panel": 312,
        "sagittata_panel": 300,
    }
    assert set(gate["strata"]) == set(expected_n)
    for name, stratum in gate["strata"].items():
        for ne_mode in ("true", "estimated"):
            result = stratum[ne_mode]
            assert result["median_100kb_pearson"] >= result["threshold"]
            assert result["n_shards"] == [expected_n[name]] * 7


def test_structured_design_realizes_preregistered_sample_counts() -> None:
    design = json.loads(STRUCTURED_DESIGN.read_text())
    assert design["uses_cross_map"] is False
    assert design["splits"]["train"]["n_shards"] == 16_000
    assert design["splits"]["val"]["n_shards"] == 1_600
    assert design["splits"]["audit"]["n_shards"] == 1_600
    assert sum(design["splits"]["train"]["design_counts"].values()) == 16_000
    assert sum(design["splits"]["audit"]["design_counts"].values()) == 1_600
    assert 0.70 <= design["splits"]["train"]["selfing"]["min"]
    assert design["splits"]["train"]["selfing"]["max"] <= 0.997
    assert set(design["splits"]["train"]["n_haplotype_counts"]) == {
        "8", "12", "16", "24", "25", "32", "50"
    }


def test_structured_completion_manifest_verifies_local_artifacts() -> None:
    completion = json.loads(STRUCTURED_COMPLETION.read_text())
    assert completion["complete"] is True
    assert completion["simulation_gate_passed"] is True
    assert completion["arabis_cross_map_used_for_model_selection"] is False
    assert completion["map_counts"] == {
        "ensemble": 32,
        **{f"seed{i}": 32 for i in range(7)},
    }
    hashes = completion["artifact_sha256"]
    assert hashes["cross_results"] == sha256(STRUCTURED_SNAPSHOT)
    assert hashes["cross_windows"] == sha256(STRUCTURED_ARRAYS)
    assert hashes["frozen_ensemble"] == sha256(STRUCTURED_MANIFEST)
    assert hashes["simulation_gate"] == sha256(STRUCTURED_GATE)
    assert hashes["simulation_design_audit"] == sha256(STRUCTURED_DESIGN)


def test_structured_correlations_rederive_and_rescue_nemorensis() -> None:
    baseline = json.loads(SNAPSHOT.read_text())
    smalln = json.loads(SMALLN_SNAPSHOT.read_text())
    summary = json.loads(STRUCTURED_SNAPSHOT.read_text())
    arrays = np.load(STRUCTURED_ARRAYS, allow_pickle=False)
    cross = np.concatenate([arrays[f"chr{i}_cross"] for i in range(1, 9)])
    primary = summary["resolutions"]["2Mb"]
    for key in ("nemorensis", "sagittata", "consensus"):
        pred = np.concatenate([arrays[f"chr{i}_{key}"] for i in range(1, 9)])
        assert np.isclose(
            spearmanr(cross, pred).statistic,
            primary["maps"][key]["spearman"],
            atol=1e-12,
        )
    assert primary["maps"]["nemorensis"]["spearman"] > baseline["resolutions"]["2Mb"]["maps"]["nemorensis"]["spearman"]
    assert primary["maps"]["nemorensis"]["spearman"] > smalln["resolutions"]["2Mb"]["maps"]["nemorensis"]["spearman"]
    assert primary["maps"]["sagittata"]["spearman"] > 0.65
    parent = summary["parent_excluded_2Mb"]["maps"]["nemorensis"]
    assert parent["spearman"] > 0.35
    assert parent["spearman_chromosome_bootstrap_ci95"][0] > 0


def test_arabis_window_audit_is_cross_map_blind_and_complete() -> None:
    audit = json.loads(WINDOW_DIAGNOSTICS.read_text())
    assert audit["f2_rate_arrays_read"] is False
    assert audit["ensemble_members"] == 7
    assert "no threshold uses the F2 map" in audit["decision_rule"]
    assert len(audit["windows"]) == 204
    assert all(len(row["seed_relative_rates"]) == 7 for row in audit["windows"])
    assert {row["species"] for row in audit["windows"]} == {
        "nemorensis",
        "sagittata",
    }


def test_arabis_visible_peaks_have_declared_support_status() -> None:
    audit = json.loads(WINDOW_DIAGNOSTICS.read_text())

    def row(species: str, chromosome: str, center_mb: float) -> dict:
        return next(
            item
            for item in audit["windows"]
            if item["species"] == species
            and item["chromosome"] == chromosome
            and item["center_mb"] == center_mb
        )

    nem_chr1 = row("nemorensis", "chr1", 5.0)
    sag_chr1 = row("sagittata", "chr1", 5.0)
    assert nem_chr1["seed_instability_extreme"] is True
    assert sag_chr1["technical_support_flag"] is False
    assert min(nem_chr1["seed_relative_rates"]) > 1.0
    assert min(sag_chr1["seed_relative_rates"]) > 1.0
    for species in ("nemorensis", "sagittata"):
        for center in (25.0, 27.0):
            assert row(species, "chr3", center)["technical_support_flag"] is False
