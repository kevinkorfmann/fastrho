"""Build the small, plot-ready data downloads used by the documentation.

The manuscript figures use compact NumPy and JSON archives.  This exporter
turns their empirical map products into stable gzipped TSV tables and bundles
the tables with the committed result snapshots.  Outputs are deterministic so
tests can compare regenerated files byte for byte.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "data" / "downloads"
PHASE2 = ROOT / "paper" / "anopheles_variants" / "phase2"
PHASE2_RELEASE = PHASE2 / "release" / "atlas_anopheles"
PHASE2_RESULTS = PHASE2 / "results"
PHASE2_MAPS = PHASE2 / "maps"
DEMOGRAPHY_PREDICTIONS = ROOT / "paper" / "figdata" / "demography_matched_predictions.npz"
SUPPLEMENTAL_RESOURCES = {
    "demography_matched_inputs.zip": (
        DEFAULT_OUTPUT / "demography_matched_inputs.zip",
        "Frozen bottleneck and expansion inputs for the paired demographic benchmark.",
    ),
    "demography_matched_results.json": (
        ROOT / "paper" / "results_snapshot" / "demography_matched.json",
        "Paired constant-history and demography-matched ReLERNN and pyrho results.",
    ),
}
RESTRICTED_ANOPHELES_NAMES = ("agam", "ag3", "arabiensis")
RESTRICTED_ANOPHELES_TEXT = (b"Ag3", b"Ag1000G phase 3", b"Aarabiensis")


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return "" if not np.isfinite(value) else f"{value:.10g}"
    return str(value)


def _tsv_bytes(columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(tuple(_cell(value) for value in row) for row in rows)
    payload = text.getvalue().encode("utf-8")
    compressed = io.BytesIO()
    with gzip.GzipFile(filename="", fileobj=compressed, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return compressed.getvalue()


def _anopheles() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    manifest_path = PHASE2_RELEASE / "manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        metadata = {row["cohort"]: row for row in csv.DictReader(handle, delimiter="\t")}
    validation = json.loads((PHASE2_RESULTS / "phase2_2la.json").read_text())
    two_la = {row["pop"]: row for row in validation["rows"]}
    ne_by_map: dict[tuple[str, str], float] = {}

    columns = (
        "cohort",
        "release_population",
        "species",
        "country",
        "chromosome_arm",
        "start_bp",
        "end_bp",
        "rate_per_bp",
        "cM_per_Mb",
        "rho_per_bp",
        "n_haplotypes",
        "Ne_used",
        "panel_twoLa_frequency",
        "panel_expected_heterokaryotype_frequency",
        "full_twoLa_frequency",
        "full_expected_heterokaryotype_frequency",
        "n_full_twoLa_samples",
    )
    rows: list[tuple[object, ...]] = []
    for path in sorted((PHASE2_RELEASE / "bed").glob("*.bed")):
        cohort = path.stem
        meta = metadata[cohort]
        arrangement = two_la[cohort]
        with path.open(encoding="utf-8", newline="") as handle:
            records = csv.DictReader(
                (line for line in handle if not line.startswith("#")),
                delimiter="\t",
                fieldnames=("chrom", "start", "end", "rate", "cm", "rho"),
            )
            for record in records:
                arm = record["chrom"]
                map_key = (cohort, arm)
                if map_key not in ne_by_map:
                    sidecar = PHASE2_MAPS / f"{cohort}__{arm}.json"
                    if not sidecar.is_file():
                        raise FileNotFoundError(
                            f"Missing arm-specific scale metadata for {cohort}/{arm}: {sidecar}"
                        )
                    ne_by_map[map_key] = float(json.loads(sidecar.read_text())["Ne_est"])
                ne_used = ne_by_map[map_key]
                rate = float(record["rate"])
                cm_per_mb = float(record["cm"])
                rho = float(record["rho"])
                # The committed BED is rounded for distribution.  Check that its three rate
                # representations still obey the declared diploid scaling contract.
                if not np.isclose(rate, rho / (4.0 * ne_used), rtol=2e-5, atol=0.0):
                    raise ValueError(f"Inconsistent rho/r/Ne scale for {cohort}/{arm}")
                if not np.isclose(cm_per_mb, rate * 1e8, rtol=5e-4, atol=5e-5):
                    raise ValueError(f"Inconsistent r/cM scale for {cohort}/{arm}")
                rows.append(
                    (
                        cohort,
                        meta["release_population"],
                        meta["species"],
                        meta["country"],
                        arm,
                        int(record["start"]),
                        int(record["end"]),
                        rate,
                        cm_per_mb,
                        rho,
                        int(meta["n_hap"]),
                        ne_used,
                        float(arrangement["panel_la_freq"]),
                        float(arrangement["panel_het_expected"]),
                        float(arrangement["la_freq"]),
                        float(arrangement["het_expected"]),
                        int(arrangement["n_samples"]),
                    )
                )
    return columns, rows


def _phase2_2la() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    data = json.loads((PHASE2_RESULTS / "phase2_2la.json").read_text())
    columns = (
        "cohort",
        "release_population",
        "species",
        "country",
        "n_full_samples",
        "n_map_samples",
        "full_twoLa_frequency",
        "full_expected_heterokaryotype_frequency",
        "full_observed_heterokaryotype_frequency",
        "panel_twoLa_frequency",
        "panel_expected_heterokaryotype_frequency",
        "panel_observed_heterokaryotype_frequency",
        "inside_outside_rate_ratio",
        "suppression_depth",
    )
    rows = [
        (
            row["pop"],
            row["release_population"],
            row["taxon"],
            row["country"],
            row["n_samples"],
            row["n_panel"],
            row["la_freq"],
            row["het_expected"],
            row["het_observed"],
            row["panel_la_freq"],
            row["panel_het_expected"],
            row["panel_het_observed"],
            row["suppression_ratio"],
            row["suppression_depth"],
        )
        for row in data["rows"]
    ]
    return columns, rows


def _phase2_resistance() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    data = json.loads((PHASE2_RESULTS / "phase2_resistance.json").read_text())
    panel = data["panels"]["hancock_mechanisms"]
    columns = (
        "cohort",
        "species",
        "resistance_region",
        "chromosome_arm",
        "position_mb",
        "focal_rate_per_bp",
        "matched_control_median_rate_per_bp",
        "focal_control_ratio",
        "nucleotide_diversity",
        "H12",
        "n_snps",
        "n_matched_controls",
        "population_panel_median_ratio",
        "population_permutation_p",
    )
    rows: list[tuple[object, ...]] = []
    for population in panel["rows"]:
        for locus, record in population["loci"].items():
            target = record["target"]
            rows.append(
                (
                    population["cohort"],
                    population["species"],
                    locus,
                    target["arm"],
                    target["mb"],
                    target["rate"],
                    record["matched_control_median_rate"],
                    record["ratio"],
                    target["pi"],
                    target["h12"],
                    target["n_snp"],
                    record["matching"]["n_matched"],
                    population["ratio"],
                    population["perm_p"],
                )
            )
    return columns, rows


def _phase2_pedigree() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    path = PHASE2_RESULTS / "pedigree" / "phase2_pedigree_windows.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    columns = tuple(records[0])
    rows = [tuple(record[column] for column in columns) for record in records]
    return columns, rows


def _phase2_pyrho() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    data = json.loads((PHASE2_RESULTS / "phase2_pyrho.json").read_text())
    columns = (
        "cohort",
        "chromosome_arm",
        "region_mb",
        "n_haplotypes",
        "n_snps",
        "watterson_Ne",
        "spearman_matched",
        "spearman_matched_p",
        "n_matched_windows",
        "spearman_published",
        "spearman_published_p",
        "n_published_windows",
    )
    rows = []
    for row in data["rows"]:
        matched = row["spearman_matched"]
        published = row["spearman_published"]
        rows.append(
            (
                row["cohort"],
                row["arm"],
                row["region_mb"],
                row["n_hap"],
                row["n_snp"],
                row["watterson_ne"],
                matched[0],
                matched[1],
                matched[2] if len(matched) > 2 else None,
                published[0],
                published[1],
                published[2] if len(published) > 2 else None,
            )
        )
    return columns, rows


def _arabis() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    campaigns = {
        "baseline_selfing": ROOT / "paper" / "figdata" / "arabis_cross.npz",
        "small_panel_selfing": ROOT / "paper" / "figdata" / "arabis_cross_smalln.npz",
        "structured_selfing": ROOT / "paper" / "figdata" / "arabis_cross_structured.npz",
    }
    columns = (
        "campaign",
        "panel_scope",
        "chromosome",
        "start_bp",
        "end_bp",
        "cross_relative_rate",
        "nemorensis_relative_rate",
        "sagittata_relative_rate",
        "consensus_relative_rate",
    )
    rows: list[tuple[object, ...]] = []
    for campaign, path in campaigns.items():
        with np.load(path) as bundle:
            for scope, prefix in (("full_panel", ""), ("parent_excluded", "parent_excluded_")):
                for chromosome in range(1, 9):
                    stem = f"{prefix}chr{chromosome}_"
                    edges = np.asarray(bundle[stem + "edges"], dtype=float)
                    series = {
                        name: np.asarray(bundle[stem + name], dtype=float)
                        for name in ("cross", "nemorensis", "sagittata", "consensus")
                    }
                    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
                        rows.append(
                            (
                                campaign,
                                scope,
                                f"chr{chromosome}",
                                int(start),
                                int(end),
                                series["cross"][index],
                                series["nemorensis"][index],
                                series["sagittata"][index],
                                series["consensus"][index],
                            )
                        )
    return columns, rows


def _arabidopsis() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    columns = (
        "chromosome",
        "center_mb",
        "fastrho_selfing_rate_per_bp",
        "pyrho_panmictic_rate_per_bp",
        "salome_meiotic_rate_per_bp",
        "rowan_meiotic_rate_per_bp",
        "n_haplotypes",
        "n_snps",
    )
    rows: list[tuple[object, ...]] = []
    with (
        np.load(ROOT / "paper" / "figdata" / "selfer_chroms.npz") as maps,
        np.load(ROOT / "paper" / "figdata" / "rowan_map.npz") as rowan,
    ):
        for chromosome in range(1, 6):
            prefix = f"c{chromosome}_"
            centers = np.asarray(maps[prefix + "centers"], dtype=float)
            series = (
                np.asarray(maps[prefix + "pred"], dtype=float),
                np.asarray(maps[prefix + "pyrho"], dtype=float),
                np.asarray(maps[prefix + "truth"], dtype=float),
                np.asarray(rowan[prefix + "rowan_100kb"], dtype=float),
            )
            if not all(values.shape == centers.shape for values in series):
                raise ValueError(f"Arabidopsis chromosome {chromosome} arrays do not align")
            for values in zip(centers, *series):
                rows.append(
                    (
                        f"chr{chromosome}",
                        *values,
                        int(maps[prefix + "nhap"]),
                        int(maps[prefix + "nsnp"]),
                    )
                )
    return columns, rows


def _mean_over_edges(edges: np.ndarray, source_edges: np.ndarray, rates: np.ndarray) -> np.ndarray:
    widths = np.diff(source_edges)
    cumulative = np.r_[0.0, np.cumsum(widths * rates)]
    integrated = np.interp(edges, source_edges, cumulative)
    return np.diff(integrated) / np.diff(edges)


def _redpoll() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    columns = (
        "chromosome",
        "start_bp",
        "end_bp",
        "center_bp",
        "pooled_rho_per_bp",
        "arrangement_A_rho_per_bp",
        "arrangement_B_rho_per_bp",
        "arrangement_A_half1_rho_per_bp",
        "arrangement_A_half2_rho_per_bp",
        "arrangement_B_half1_rho_per_bp",
        "arrangement_B_half2_rho_per_bp",
        "inside_supergene",
    )
    with (
        np.load(ROOT / "paper" / "figdata" / "fieldguide_redpoll.npz") as pooled,
        np.load(ROOT / "paper" / "figdata" / "redpoll_karyotype_maps.npz") as groups,
    ):
        edges = np.asarray(groups["edges"], dtype=float)
        source_edges = np.r_[float(pooled["pos_left"][0]), np.asarray(pooled["pos_right"], float)]
        pooled_rate = _mean_over_edges(edges, source_edges, np.asarray(pooled["rho_per_bp"], float))
        inv_start, inv_end = int(groups["inv_start"]), int(groups["inv_end"])
        names = (
            "arrangement_A_rate",
            "arrangement_B_rate",
            "arrangement_A_half1_rate",
            "arrangement_A_half2_rate",
            "arrangement_B_half1_rate",
            "arrangement_B_half2_rate",
        )
        series = [np.asarray(groups[name], dtype=float) for name in names]
        rows = []
        for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
            center = float(groups["centers"][index])
            rows.append(
                (
                    "chr1",
                    int(start),
                    int(end),
                    center,
                    pooled_rate[index],
                    *(values[index] for values in series),
                    center >= inv_start and center < inv_end,
                )
            )
    return columns, rows


def _tree_of_life() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    columns = (
        "species_key",
        "common_name",
        "latin_name",
        "clade",
        "source",
        "model",
        "status",
        "qualification_tier",
        "qualification_note",
        "validated",
        "center_mb",
        "predicted_rate_per_bp",
        "reference_rate_per_bp",
        "reference_map",
        "n_diploids",
        "n_haplotypes",
    )
    data = json.loads((ROOT / "paper" / "figdata" / "transect.json").read_text())
    rows: list[tuple[object, ...]] = []
    for species in data["species"]:
        track = species.get("track") or {}
        centers = track.get("centers") or []
        predicted = track.get("pred") or []
        reference = track.get("truth") or [None] * len(centers)
        if len(reference) != len(centers):
            reference = [None] * len(centers)
        if len(predicted) != len(centers):
            raise ValueError(f"tree-of-life track mismatch for {species['key']}")
        for center, prediction, truth in zip(centers, predicted, reference):
            rows.append(
                (
                    species["key"],
                    species["common"],
                    species["latin"],
                    species["clade"],
                    species["source"],
                    species["model"],
                    species["status"],
                    species.get("qualification_tier", "core"),
                    species.get("qualification_note"),
                    species["validated"],
                    center,
                    prediction,
                    truth,
                    species.get("map_label"),
                    species["n_dip"],
                    species["n_hap"],
                )
            )
    return columns, rows


def _canid_example() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    columns = (
        "center_mb",
        "generative_rate_per_bp",
        "large_population_inferred_rate_per_bp",
        "bottlenecked_population_inferred_rate_per_bp",
    )
    with np.load(ROOT / "paper" / "figdata" / "dog_fig.npz") as data:
        rows = list(
            zip(
                np.asarray(data["b_centers"], float),
                np.asarray(data["b_truth"], float),
                np.asarray(data["b_vil"], float),
                np.asarray(data["b_brd"], float),
            )
        )
    return columns, rows


def _canid_recovery() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    columns = (
        "region_id",
        "bottlenecked_population_Ne",
        "own_data_log_pearson",
        "source_transfer_log_pearson",
    )
    with np.load(ROOT / "paper" / "figdata" / "dog_fig.npz") as data:
        rows = [
            (index, ne, own, transfer)
            for index, (ne, own, transfer) in enumerate(zip(data["Ne"], data["own"], data["trn"]))
        ]
    return columns, rows


def _demography_matched() -> tuple[tuple[str, ...], list[tuple[object, ...]]]:
    columns = (
        "scenario",
        "method",
        "history",
        "region_id",
        "start_bp",
        "end_bp",
        "true_rate_per_bp",
        "predicted_rate_per_bp",
    )
    rows: list[tuple[object, ...]] = []
    with np.load(DEMOGRAPHY_PREDICTIONS) as data:
        metadata = json.loads(str(data["_metadata"]))
        grid = int(metadata["grid_bp"])
        for arm, record in sorted(metadata["arms"].items()):
            scenario = record["scenario"]
            region_keys = sorted(
                key.removeprefix(f"pred__{arm}__")
                for key in data.files
                if key.startswith(f"pred__{arm}__")
            )
            if len(region_keys) != record["n_regions"]:
                raise ValueError(f"Prediction archive is incomplete for {arm}")
            for region in region_keys:
                truth = np.asarray(data[f"truth__{scenario}__{region}"], dtype=float)
                prediction = np.asarray(data[f"pred__{arm}__{region}"], dtype=float)
                if truth.shape != prediction.shape:
                    raise ValueError(f"Prediction archive is misaligned for {arm}/{region}")
                for index, (true_rate, predicted_rate) in enumerate(zip(truth, prediction)):
                    rows.append(
                        (
                            scenario,
                            record["method"],
                            record["history"],
                            region,
                            index * grid,
                            (index + 1) * grid,
                            true_rate,
                            predicted_rate,
                        )
                    )
    return columns, rows


DATASETS = {
    "anopheles_maps.tsv.gz": (
        _anopheles,
        "Nine open Ag1000G Phase 2 AR1 population maps across five chromosome arms at 50-kb resolution.",
        "50-kb, 0-based half-open AgamP4 windows; rho_per_bp is population scaled; rate_per_bp = rho_per_bp / (4 * Ne_used); cM_per_Mb = rate_per_bp * 1e8; Ne_used is the arm-specific auxiliary model estimate; 2La columns distinguish the 40-mosquito map panel from all eligible released samples",
        [
            "paper/anopheles_variants/phase2/release/atlas_anopheles/bed",
            "paper/anopheles_variants/phase2/release/atlas_anopheles/manifest.tsv",
            "paper/anopheles_variants/phase2/maps/*__*.json",
            "paper/anopheles_variants/phase2/results/phase2_2la.json",
        ],
    ),
    "phase2_2la.tsv.gz": (
        _phase2_2la,
        "Population-level 2La arrangement frequencies and inferred suppression summaries for the nine Phase 2 cohorts.",
        "frequencies, inside/outside rate ratio, and suppression depth",
        ["paper/anopheles_variants/phase2/results/phase2_2la.json"],
    ),
    "phase2_resistance.tsv.gz": (
        _phase2_resistance,
        "Focal and diversity/H12-matched control rates for the prespecified 15 resistance regions in all nine Phase 2 populations.",
        "rate per bp, focal/control ratio, nucleotide diversity, H12, and matching metadata",
        ["paper/anopheles_variants/phase2/results/phase2_resistance.json"],
    ),
    "phase2_pedigree_windows.tsv.gz": (
        _phase2_pedigree,
        "Broad-scale Phase 2 laboratory-cross and inferred-atlas rates on common 5-Mb autosomal windows.",
        "within-arm normalized direct and inferred rates",
        ["paper/anopheles_variants/phase2/results/pedigree/phase2_pedigree_windows.tsv"],
    ),
    "phase2_pyrho.tsv.gz": (
        _phase2_pyrho,
        "Matched-window concordance between fastrho and pyrho for three frozen Phase 2 regions.",
        "Spearman correlation, two-sided p-value, and common-window count",
        ["paper/anopheles_variants/phase2/results/phase2_pyrho.json"],
    ),
    "arabis_cross_maps.tsv.gz": (
        _arabis,
        "F2 cross and inferred recombination-map shapes for both Arabis panels and three campaigns.",
        "chromosome-relative rate; each chromosome and series has mean 1",
        [
            "paper/figdata/arabis_cross.npz",
            "paper/figdata/arabis_cross_smalln.npz",
            "paper/figdata/arabis_cross_structured.npz",
        ],
    ),
    "arabidopsis_maps.tsv.gz": (
        _arabidopsis,
        "Selfing-aware, panmictic, and two meiotic Arabidopsis thaliana maps.",
        "rate per bp at 100-kb resolution",
        ["paper/figdata/selfer_chroms.npz", "paper/figdata/rowan_map.npz"],
    ),
    "redpoll_maps.tsv.gz": (
        _redpoll,
        "Pooled and arrangement-stratified redpoll chromosome-1 maps.",
        "population-scaled rho per bp at 500-kb resolution",
        [
            "paper/figdata/fieldguide_redpoll.npz",
            "paper/figdata/redpoll_karyotype_maps.npz",
        ],
    ),
    "tree_of_life_maps.tsv.gz": (
        _tree_of_life,
        "Representative 100-kb tracks for the ten-species cross-species comparison.",
        "rate per bp; status distinguishes validation and repeatability-qualified tracks; qualification_tier distinguishes core and context-limited cohorts",
        ["paper/figdata/transect.json"],
    ),
    "canid_example_map.tsv.gz": (
        _canid_example,
        "One shared simulated canid landscape under large and bottlenecked populations.",
        "rate per bp at 50-kb resolution",
        ["paper/figdata/dog_fig.npz"],
    ),
    "canid_recovery.tsv.gz": (
        _canid_recovery,
        "Paired canid own-data and source-transfer recovery over 120 regions.",
        "100-kb log-Pearson correlation",
        ["paper/figdata/dog_fig.npz"],
    ),
}

if DEMOGRAPHY_PREDICTIONS.is_file():
    DATASETS["demography_matched_windows.tsv.gz"] = (
        _demography_matched,
        "Every paired ReLERNN and pyrho prediction for the bottleneck and expansion benchmark.",
        "rate per bp on the common 25-kb grid",
        ["paper/figdata/demography_matched_predictions.npz"],
    )


def _zip_entry(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, payload)


def _assert_public_anopheles_payload(name: str, payload: bytes) -> None:
    lowered = name.lower()
    if any(marker in lowered for marker in RESTRICTED_ANOPHELES_NAMES):
        raise ValueError(f"restricted Anopheles artifact cannot be bundled: {name}")
    if any(marker in payload for marker in RESTRICTED_ANOPHELES_TEXT):
        raise ValueError(f"restricted Anopheles result text cannot be bundled: {name}")


def _phase2_result_bundle() -> bytes:
    sources = (
        PHASE2 / "config.json",
        PHASE2_RELEASE / "manifest.tsv",
        PHASE2_RELEASE / "provenance.json",
        PHASE2_RESULTS / "phase2_2la.json",
        PHASE2_RESULTS / "phase2_map_qc.json",
        PHASE2_RESULTS / "phase2_pyrho.json",
        PHASE2_RESULTS / "phase2_resistance.json",
        PHASE2_RESULTS / "pedigree" / "phase2_pedigree.json",
        PHASE2_RESULTS / "pedigree" / "phase2_pedigree_windows.tsv",
    )
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w") as archive:
        for source in sources:
            name = source.relative_to(PHASE2).as_posix()
            payload = source.read_bytes()
            _assert_public_anopheles_payload(name, payload)
            _zip_entry(archive, name, payload)
    return bundle.getvalue()


ANOPHELES_README = """fastrho Phase 2 mosquito maps
================================

Contents
--------
anopheles_maps.tsv.gz contains 50-kb population recombination-map windows for
nine open Ag1000G Phase 2 AR1 cohorts, five AgamP4 chromosome arms, and the
fixed 40-diploid inference panels.  It also carries cohort/species metadata and
panel-versus-full-release 2La summaries.

Coordinates and scale
---------------------
start_bp/end_bp are 0-based, half-open AgamP4 coordinates.  rho_per_bp is the
population-scaled output.  rate_per_bp and cM_per_Mb are already placed on an
absolute scale using the arm-specific auxiliary model estimate in Ne_used:

    rate_per_bp = rho_per_bp / (4 * Ne_used)
    cM_per_Mb   = rate_per_bp * 1e8

Do not divide rate_per_bp or cM_per_Mb by Ne again.  No rescaling is needed to
plot the released values.  If an independently justified effective population
size Ne_target is preferred, recompute the conditional absolute columns from
rho_per_bp as rho_per_bp / (4 * Ne_target) and multiply by 1e8 for cM/Mb.
Ne_used is an auxiliary model point estimate, not an independently validated
demographic or census estimate.  Absolute comparisons therefore remain
conditional on the selected Ne; retain rho_per_bp for the released
population-scaled quantity.

Verification
------------
The parent documentation download manifest records this archive's SHA-256 and
the table's row count, columns, source artifacts, byte size, and SHA-256.
Regenerate it with: python scripts/export_paper_data.py
"""


def _anopheles_download_bundle(table: bytes) -> bytes:
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w") as archive:
        _zip_entry(archive, "README.txt", ANOPHELES_README.encode("utf-8"))
        _zip_entry(archive, "anopheles_maps.tsv.gz", table)
    return bundle.getvalue()


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    generated: dict[str, bytes] = {}
    datasets = []
    for filename, (loader, description, units, sources) in DATASETS.items():
        columns, rows = loader()
        payload = _tsv_bytes(columns, rows)
        generated[filename] = payload
        datasets.append(
            {
                "file": filename,
                "description": description,
                "units": units,
                "rows": len(rows),
                "columns": list(columns),
                "source_artifacts": sources,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )

    resources = []
    resource_payloads = {
        "phase2_results.zip": _phase2_result_bundle(),
        "anopheles_maps.zip": _anopheles_download_bundle(generated["anopheles_maps.tsv.gz"]),
    }
    resources.append(
        {
            "file": "phase2_results.zip",
            "description": "Compact source results, cohort manifest, and provenance for the active Phase 2 manuscript analysis.",
            "sha256": hashlib.sha256(resource_payloads["phase2_results.zip"]).hexdigest(),
            "bytes": len(resource_payloads["phase2_results.zip"]),
        }
    )
    resources.append(
        {
            "file": "anopheles_maps.zip",
            "description": "Self-documenting mosquito-map download containing the complete table and an explicit coordinate, scale, and Ne-rescaling contract.",
            "sha256": hashlib.sha256(resource_payloads["anopheles_maps.zip"]).hexdigest(),
            "bytes": len(resource_payloads["anopheles_maps.zip"]),
        }
    )
    for filename, (source, description) in SUPPLEMENTAL_RESOURCES.items():
        if not source.is_file():
            continue
        payload = source.read_bytes()
        resource_payloads[filename] = payload
        resources.append(
            {
                "file": filename,
                "description": description,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )

    manifest = {
        "schema_version": 1,
        "description": "Plot-ready empirical maps and map-recovery outputs from the fastrho paper.",
        "datasets": datasets,
        "resources": resources,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    readme = (
        "fastrho paper data\n"
        "===================\n\n"
        "Each table is a UTF-8, tab-delimited file compressed with gzip. Blank cells are "
        "missing values. Coordinates are 0-based half-open where start/end columns are present.\n\n"
        "The active Anopheles analysis uses the open Ag1000G Phase 2 AR1 release: nine "
        "population panels of 40 mosquitoes across five chromosome arms. The map table labels "
        "2La summaries from both the map panels and all eligible released samples. Its "
        "rho_per_bp column is population-scaled; rate_per_bp and cM_per_Mb are already "
        "converted with the arm-specific auxiliary estimate in Ne_used and must not be divided "
        "by Ne again. See resources/anopheles_maps.zip for the complete rescaling contract.\n\n"
        "The tables are plot-ready exports of the compact manuscript artifacts. See manifest.json "
        "for schemas, units, row counts, source artifacts, and SHA-256 checksums. The results/ "
        "directory contains the committed JSON snapshots behind reported paper metrics.\n"
    ).encode()

    for filename, payload in generated.items():
        (output / filename).write_bytes(payload)
    for filename, payload in resource_payloads.items():
        (output / filename).write_bytes(payload)
    (output / "manifest.json").write_bytes(manifest_bytes)

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w") as archive:
        _zip_entry(archive, "README.txt", readme)
        _zip_entry(archive, "manifest.json", manifest_bytes)
        for filename, payload in sorted(generated.items()):
            _zip_entry(archive, f"tables/{filename}", payload)
        for filename, payload in sorted(resource_payloads.items()):
            _zip_entry(archive, f"resources/{filename}", payload)
        snapshot_root = ROOT / "paper" / "results_snapshot"
        for path in sorted(snapshot_root.rglob("*.json")):
            relative = path.relative_to(snapshot_root).as_posix()
            payload = path.read_bytes()
            _assert_public_anopheles_payload(relative, payload)
            _zip_entry(archive, f"results/{relative}", payload)
    (output / "fastrho_paper_data.zip").write_bytes(bundle.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
