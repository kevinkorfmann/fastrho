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
AG3 = ROOT / "paper" / "anopheles_variants" / "ag3"
AG3_RELEASE = AG3 / "release" / "atlas_anopheles"
AG3_MAPS = AG3 / "maps"
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
    manifest_path = AG3_RELEASE / "manifest.tsv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        metadata = {row["cohort"]: row for row in csv.DictReader(handle, delimiter="\t")}

    columns = (
        "cohort",
        "species",
        "country",
        "latitude",
        "longitude",
        "chromosome_arm",
        "start_bp",
        "end_bp",
        "rate_per_bp",
        "cM_per_Mb",
        "rho_per_bp",
        "n_haplotypes",
        "n_snps",
        "Ne_used",
        "panel_twoLa_frequency",
        "panel_expected_heterokaryotype_frequency",
    )
    rows: list[tuple[object, ...]] = []
    map_metadata: dict[tuple[str, str], tuple[float, int]] = {}
    for path in sorted((AG3_RELEASE / "bed").glob("*.bed")):
        cohort = path.stem
        meta = metadata[cohort]
        with path.open(encoding="utf-8", newline="") as handle:
            records = csv.DictReader(
                (line for line in handle if not line.startswith("#")),
                delimiter="\t",
                fieldnames=("chrom", "start", "end", "rate", "cm", "rho"),
            )
            for record in records:
                arm = record["chrom"]
                map_key = (cohort, arm)
                if map_key not in map_metadata:
                    with np.load(AG3_MAPS / f"{cohort}__{arm}.npz") as map_data:
                        map_metadata[map_key] = (
                            float(map_data["Ne_est"]),
                            int(map_data["n_snp"]),
                        )
                ne_used, n_snps = map_metadata[map_key]
                rho = float(record["rho"])
                rate = rho / (4.0 * ne_used)
                cm_per_mb = rate * 1e8
                rows.append(
                    (
                        cohort,
                        meta["species"],
                        meta["country"],
                        float(meta["lat"]),
                        float(meta["lon"]),
                        arm,
                        int(record["start"]),
                        int(record["end"]),
                        rate,
                        cm_per_mb,
                        rho,
                        int(meta["n_hap"]),
                        n_snps,
                        ne_used,
                        float(meta["twoLa_p"]),
                        float(meta["twoLa_H"]),
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
        "Thirteen MalariaGEN Ag3.0 population maps across five chromosome arms at 50-kb resolution.",
        "50-kb, 0-based half-open AgamP4 windows; rho_per_bp is population scaled; rate_per_bp = rho_per_bp / (4 * Ne_used); cM_per_Mb = rate_per_bp * 1e8; Ne_used is the arm-specific auxiliary model estimate; 2La columns describe the fixed 40-mosquito inference panel",
        [
            "paper/anopheles_variants/ag3/release/atlas_anopheles/bed",
            "paper/anopheles_variants/ag3/release/atlas_anopheles/manifest.tsv",
            "paper/anopheles_variants/ag3/maps/*__*.npz",
        ],
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


ANOPHELES_README = """fastrho Ag3.0 mosquito maps
============================

Contents
--------
anopheles_maps.tsv.gz contains 50-kb population recombination-map windows for
13 MalariaGEN Ag3.0 cohorts from Anopheles gambiae, An. coluzzii, and
An. arabiensis, across five AgamP4 chromosome arms. Every cohort uses a fixed
40-diploid inference panel, retained across all five arms.

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
        "anopheles_maps.zip": _anopheles_download_bundle(generated["anopheles_maps.tsv.gz"]),
    }
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
        "Map scales\n"
        "----------\n"
        "anopheles_maps.tsv.gz: rho_per_bp is population scaled; rate_per_bp and cM_per_Mb "
        "use the recorded Ne_used.\n"
        "arabis_cross_maps.tsv.gz: dimensionless chromosome-relative rates with mean 1 within "
        "each map series.\n"
        "arabidopsis_maps.tsv.gz: per-generation rates per bp at 100-kb resolution.\n"
        "redpoll_maps.tsv.gz: population-scaled rho per bp at 500-kb resolution.\n"
        "tree_of_life_maps.tsv.gz: per-generation rates per bp at 100-kb resolution; normalize "
        "within species for cross-species shape comparisons.\n"
        "canid_example_map.tsv.gz: per-generation rates per bp for the simulated landscape and "
        "both inferred maps.\n\n"
        "A column ending in _relative_rate is dimensionless and cannot be converted to cM/Mb. "
        "For a per-generation rate_per_bp column, cM/Mb = rate_per_bp * 1e8. Only the mosquito "
        "table records the Ne used for its absolute conversion. Treat inferred per-bp values in "
        "the Arabidopsis and tree-of-life tables as the archived paper scale for within-track "
        "comparisons, not as independently calibrated cross-species absolute rates.\n\n"
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
