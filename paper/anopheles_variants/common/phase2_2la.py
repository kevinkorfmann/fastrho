#!/usr/bin/env python3
"""Recompute the 2La graded-landmark analysis from Phase 2 haplotypes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr

BREAKPOINTS_MB = (20.524, 42.166)
WINDOW = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def suppression(maps: Path, cohort: str) -> tuple[float, float]:
    path = maps / f"{cohort}__2L.npz"
    with np.load(path, allow_pickle=True) as data:
        starts = np.asarray(data[f"starts_{WINDOW}"], dtype=float)
        rho = np.asarray(data[f"rho_{WINDOW}"], dtype=float)
    centers = (starts + WINDOW / 2) / 1e6
    valid = np.isfinite(rho) & (rho > 0)
    inside = valid & (centers >= BREAKPOINTS_MB[0]) & (centers <= BREAKPOINTS_MB[1])
    outside = valid & ~inside
    if inside.sum() < 3 or outside.sum() < 3:
        raise ValueError(f"insufficient supported windows for {cohort}")
    ratio = float(np.median(rho[inside]) / np.median(rho[outside]))
    return ratio, 1.0 - ratio


def tag_karyotypes(
    genotype: h5py.Dataset,
    positions: np.ndarray,
    refs: np.ndarray,
    alts: np.ndarray,
    tags: list[tuple[int, str]],
    sample_indices: list[int],
) -> tuple[np.ndarray, int]:
    source_indices = []
    tag_indices = []
    for position, allele in tags:
        index = int(np.searchsorted(positions, position))
        if index >= len(positions) or int(positions[index]) != position:
            continue
        ref = refs[index].decode() if isinstance(refs[index], bytes) else str(refs[index])
        alt = alts[index].decode() if isinstance(alts[index], bytes) else str(alts[index])
        if allele == ref:
            tag_index = 0
        elif allele == alt:
            tag_index = 1
        else:
            continue
        source_indices.append(index)
        tag_indices.append(tag_index)
    if not source_indices:
        raise ValueError("no Phase 2 2La tag SNPs matched released biallelic haplotypes")
    sample_indices_array = np.asarray(sample_indices, dtype=int)
    order = np.argsort(sample_indices_array)
    sample_indices_array = sample_indices_array[order]
    calls = np.asarray(genotype[source_indices, :, :][:, sample_indices_array, :], dtype=np.int8)
    tag_indices_array = np.asarray(tag_indices, dtype=np.int8)[:, None, None]
    called = (calls >= 0).all(axis=2)
    dosages = (calls == tag_indices_array).sum(axis=2)
    means = np.ma.MaskedArray(dosages, mask=~called).mean(axis=0).filled(np.nan)
    inverse = np.argsort(order)
    return np.asarray(means)[inverse], len(source_indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tag-snps", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.selection.open(encoding="utf-8", newline="") as handle:
        selection = list(csv.DictReader(handle, delimiter="\t"))
    with args.selected_samples.open(encoding="utf-8", newline="") as handle:
        selected_rows = list(csv.DictReader(handle, delimiter="\t"))
    selected: dict[str, list[str]] = {}
    for row in selected_rows:
        selected.setdefault(row["cohort"], []).append(row["sample_id"])
    with args.metadata.open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    release_samples: dict[str, list[str]] = {}
    for row in metadata:
        release_samples.setdefault(row["population"], []).append(row["ox_code"])
    with args.tag_snps.open(encoding="utf-8", newline="") as handle:
        tags = [
            (int(row["position"]), row["alt_allele"])
            for row in csv.DictReader(handle)
            if row["inversion"] == "2La"
        ]

    with h5py.File(args.hdf5, "r") as source:
        group = source["2L"]
        samples = decode(group["samples"][:])
        sample_index = {sample: index for index, sample in enumerate(samples)}
        positions = np.asarray(group["variants/POS"], dtype=np.int64)
        refs = np.asarray(group["variants/REF"])
        alts = np.asarray(group["variants/ALT"])
        rows = []
        matched_count = None
        for cohort_row in selection:
            cohort = cohort_row["cohort"]
            population = cohort_row["release_population"]
            full_ids = [sample for sample in release_samples[population] if sample in sample_index]
            panel_ids = selected[cohort]
            full_mean, full_matched = tag_karyotypes(
                group["calldata/genotype"], positions, refs, alts, tags,
                [sample_index[sample] for sample in full_ids],
            )
            panel_mean, panel_matched = tag_karyotypes(
                group["calldata/genotype"], positions, refs, alts, tags,
                [sample_index[sample] for sample in panel_ids],
            )
            if full_matched != panel_matched:
                raise ValueError("tag-SNP matching differs between full and panel calls")
            matched_count = full_matched
            full_p = float(np.nanmean(full_mean) / 2)
            panel_p = float(np.nanmean(panel_mean) / 2)
            ratio, depth = suppression(args.maps, cohort)
            rows.append(
                {
                    "pop": cohort,
                    "release_population": population,
                    "taxon": cohort_row["species"],
                    "country": cohort_row["country"],
                    "n_samples": len(full_ids),
                    "n_panel": len(panel_ids),
                    "la_freq": full_p,
                    "het_expected": 2 * full_p * (1 - full_p),
                    "het_observed": float(np.nanmean(np.rint(full_mean) == 1)),
                    "panel_la_freq": panel_p,
                    "panel_het_expected": 2 * panel_p * (1 - panel_p),
                    "panel_het_observed": float(np.nanmean(np.rint(panel_mean) == 1)),
                    "suppression_ratio": ratio,
                    "suppression_depth": depth,
                    "kar_source": "Phase 2 released haplotypes and MalariaGEN 2La tag SNPs",
                }
            )

    expected = np.asarray([row["het_expected"] for row in rows])
    observed = np.asarray([row["het_observed"] for row in rows])
    depth = np.asarray([row["suppression_depth"] for row in rows])
    spearman_expected = spearmanr(expected, depth)
    pearson_expected = pearsonr(expected, depth)
    spearman_observed = spearmanr(observed, depth)
    result = {
        "schema_version": 1,
        "variant": "phase2",
        "release": "Ag1000G Phase 2 AR1",
        "rows": rows,
        "la_breakpoints_mb": list(BREAKPOINTS_MB),
        "window": WINDOW,
        "tag_snps_requested": len(tags),
        "tag_snps_matched": matched_count,
        "spearman_Hexp_depth": [float(spearman_expected.statistic), float(spearman_expected.pvalue)],
        "pearson_Hexp_depth": [float(pearson_expected.statistic), float(pearson_expected.pvalue)],
        "spearman_Hobs_depth": [float(spearman_observed.statistic), float(spearman_observed.pvalue)],
        "n_cohorts": len(rows),
        "provenance": {
            "hdf5_sha256": sha256(args.hdf5),
            "metadata_sha256": sha256(args.metadata),
            "selected_samples_sha256": sha256(args.selected_samples),
            "tag_snps_sha256": sha256(args.tag_snps),
            "map_sha256": {row["pop"]: sha256(args.maps / f"{row['pop']}__2L.npz") for row in rows},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)
    print("Pearson", result["pearson_Hexp_depth"])
    print("Spearman", result["spearman_Hexp_depth"])


if __name__ == "__main__":
    main()
