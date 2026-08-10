#!/usr/bin/env python3
"""Phase 2 resistance-region analysis with haplotype-matched controls.

This is intentionally release-specific.  It consumes the normalized Phase 2
HDF5 panels and the corresponding frozen-checkpoint maps, retains the
literature-defined 6/8/15-region panels, and reports population-level and
species-stratified descriptive summaries. It does not perform restricted-release
arabiensis label-enumeration or resistance-allele carrier analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

ARM_LEN = {"2R": 61.5, "2L": 49.4, "3R": 53.2, "3L": 42.0, "X": 24.4}
CEN_MAX = {"2R": True, "3R": True, "X": True, "2L": False, "3L": False}
LEVELS = {"core_six": 1, "surveillance_markers": 2, "hancock_mechanisms": 3}
LOCAL_MB = 0.15
FEATURE_MB = 0.15
EXCLUDE_MB = 0.50
H12_SNPS = 128
PRESELECT = 500
K_MATCH = 75
MIN_MATCH = 30
NPERM = 5_000


@dataclass(frozen=True)
class Locus:
    name: str
    arm: str
    mb: float
    tier: int


@dataclass(frozen=True)
class Feature:
    arm: str
    mb: float
    rate: float
    telo: float
    n_snp: int
    log_density: float
    pi: float
    h12: float


def read_loci(path: Path) -> list[Locus]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            Locus(row["locus"], row["arm"], float(row["mb"]), int(row["tier"]))
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def telo_frac(arm: str, mb: float) -> float:
    length = ARM_LEN[arm]
    return (length - mb) / length if CEN_MAX[arm] else mb / length


def h12(haps: np.ndarray) -> float:
    if haps.shape[1] == 0:
        return float("nan")
    if haps.shape[1] > H12_SNPS:
        take = np.unique(np.linspace(0, haps.shape[1] - 1, H12_SNPS).round().astype(int))
        haps = haps[:, take]
    _, counts = np.unique(np.ascontiguousarray(haps), axis=0, return_counts=True)
    freq = np.sort(counts.astype(float) / haps.shape[0])[::-1]
    if len(freq) == 1:
        return float(freq[0] ** 2)
    return float((freq[0] + freq[1]) ** 2 + np.sum(freq[2:] ** 2))


def hap_features(gm: np.ndarray, pos: np.ndarray, mb: float) -> tuple[int, float, float, float]:
    bp = mb * 1e6
    lo = np.searchsorted(pos, bp - FEATURE_MB * 1e6, side="left")
    hi = np.searchsorted(pos, bp + FEATURE_MB * 1e6, side="right")
    local = gm[:, lo:hi]
    n_snp = int(local.shape[1])
    if n_snp == 0:
        return 0, float("nan"), float("nan"), float("nan")
    freq = local.mean(axis=0)
    density = n_snp / (2 * FEATURE_MB * 1_000)
    return n_snp, float(np.log1p(density)), float(np.mean(2 * freq * (1 - freq))), h12(local)


def load_arm(normalized: Path, maps: Path, cohort: str, arm: str):
    with h5py.File(normalized / f"{cohort}__{arm}.h5", "r") as handle:
        gm = np.asarray(handle["gm"], dtype=np.int8)
        pos = np.asarray(handle["pos"], dtype=float)
    with np.load(maps / f"{cohort}__{arm}.npz", allow_pickle=True) as z:
        starts = z["starts_50000"].astype(float) / 1e6
        rates = z["r_50000"].astype(float)
    return gm, pos, starts, rates


def local_rate(starts: np.ndarray, rates: np.ndarray, mb: float) -> float:
    keep = (starts >= mb - LOCAL_MB) & (starts <= mb + LOCAL_MB) & (rates > 0)
    return float(np.median(rates[keep])) if np.any(keep) else float("nan")


def make_feature(arm: str, mb: float, rate: float, gm: np.ndarray, pos: np.ndarray) -> Feature:
    n_snp, density, pi, local_h12 = hap_features(gm, pos, mb)
    return Feature(arm, mb, rate, telo_frac(arm, mb), n_snp, density, pi, local_h12)


def percentile_distance(target: Feature, candidates: list[Feature]):
    usable = [
        row for row in candidates
        if row.rate > 0 and all(np.isfinite(x) for x in (row.log_density, row.pi, row.h12))
    ]
    if not usable:
        return [], {}
    fields = ("telo", "log_density", "pi", "h12")
    matrix = np.array([[getattr(row, key) for key in fields] for row in usable], float)
    target_values = np.array([getattr(target, key) for key in fields], float)
    ranks = np.empty_like(matrix)
    target_rank = np.empty(len(fields), float)
    for index in range(len(fields)):
        order = np.argsort(matrix[:, index])
        ranks[order, index] = (np.arange(len(usable)) + 0.5) / len(usable)
        target_rank[index] = np.searchsorted(
            np.sort(matrix[:, index]), target_values[index], side="right"
        ) / len(usable)
    distance = np.sqrt(np.sum(((ranks - target_rank) * np.array([1.5, 1, 1, 1])) ** 2, axis=1))
    chosen_index = np.argsort(distance)[: min(K_MATCH, len(usable))]
    chosen = [usable[index] for index in chosen_index]
    balance = np.median(np.abs(ranks[chosen_index] - target_rank), axis=0)
    return chosen, {
        "n_candidates": len(usable),
        "n_matched": len(chosen),
        "target_percentile": dict(zip(fields, target_rank.tolist())),
        "median_abs_percentile_delta": dict(zip(fields, balance.tolist())),
    }


def cohort_panel(
    normalized: Path,
    maps: Path,
    cohort: str,
    species: str,
    panel: list[Locus],
    exclusion: list[Locus],
    rng: np.random.Generator,
) -> dict:
    arm_cache = {}
    log_ratios = []
    permutation_sum = np.zeros(NPERM, float)
    per_locus = {}
    for locus in panel:
        if locus.arm not in arm_cache:
            arm_cache[locus.arm] = load_arm(normalized, maps, cohort, locus.arm)
        gm, pos, starts, rates = arm_cache[locus.arm]
        target = make_feature(locus.arm, locus.mb, local_rate(starts, rates, locus.mb), gm, pos)
        valid = (rates > 0) & np.isfinite(rates)
        ordered = np.argsort(np.abs(np.array([telo_frac(locus.arm, x) for x in starts[valid]]) - target.telo))
        win_mb = starts[valid]
        win_rate = rates[valid]
        candidates = []
        for index in ordered:
            mb = float(win_mb[index])
            if any(other.arm == locus.arm and abs(mb - other.mb) <= EXCLUDE_MB for other in exclusion):
                continue
            candidates.append(make_feature(locus.arm, mb, float(win_rate[index]), gm, pos))
            if len(candidates) == PRESELECT:
                break
        matched, matching = percentile_distance(target, candidates)
        control = np.array([row.rate for row in matched], float)
        if len(control) >= MIN_MATCH and np.isfinite(target.rate):
            control_median = float(np.median(control))
            ratio = target.rate / control_median
            log_ratios.append(np.log(ratio))
            draws = control[rng.integers(0, len(control), size=NPERM)]
            permutation_sum += np.log(draws / control_median)
        else:
            control_median = ratio = float("nan")
        per_locus[locus.name] = {
            "target": asdict(target),
            "matched_control_median_rate": control_median,
            "ratio": ratio,
            "matching": matching,
        }
    observed = float(np.mean(log_ratios)) if log_ratios else float("nan")
    null = permutation_sum / max(len(log_ratios), 1)
    return {
        "cohort": cohort,
        "species": species,
        "ratio": float(np.exp(observed)) if log_ratios else float("nan"),
        "perm_p": float(np.mean(null <= observed)) if log_ratios else float("nan"),
        "n_loci": len(log_ratios),
        "loci": per_locus,
    }


def summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if np.isfinite(row["ratio"])]
    by_species = {}
    for species in sorted({row["species"] for row in valid}):
        group = [row for row in valid if row["species"] == species]
        by_species[species] = {
            "n_cohorts": len(group),
            "median_ratio": float(np.median([row["ratio"] for row in group])),
            "n_nominal_p_lt_0_05": sum(row["perm_p"] < 0.05 for row in group),
        }
    return {
        "n_cohorts": len(valid),
        "median_ratio": float(np.median([row["ratio"] for row in valid])),
        "n_nominal_p_lt_0_05": sum(row["perm_p"] < 0.05 for row in valid),
        "by_species": by_species,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--panel-spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    loci = read_loci(args.panel_spec)
    with args.selection.open(newline="", encoding="utf-8") as handle:
        cohorts = list(csv.DictReader(handle, delimiter="\t"))
    result = {
        "release": "Ag1000G Phase 2 AR1",
        "panel_spec_sha256": hashlib.sha256(args.panel_spec.read_bytes()).hexdigest(),
        "design": {
            "local_mb": LOCAL_MB,
            "feature_mb": FEATURE_MB,
            "exclusion_mb": EXCLUDE_MB,
            "nperm": NPERM,
            "primary_interpretation": "aggregate across nine confirmed gambiae/coluzzii cohorts",
            "scope_note": "Only tests supported by the open Phase 2 release are included.",
        },
        "panels": {},
    }
    for panel_index, (panel_name, level) in enumerate(LEVELS.items()):
        panel = [locus for locus in loci if locus.tier <= level]
        rows = [
            cohort_panel(
                args.normalized,
                args.maps,
                row["cohort"],
                row["species"],
                panel,
                loci,
                np.random.default_rng(20_260_806 + panel_index * 100 + index),
            )
            for index, row in enumerate(cohorts)
            if row.get("include_primary", "1") in {"1", "true", "True"}
        ]
        result["panels"][panel_name] = {
            "loci": [locus.name for locus in panel],
            "rows": rows,
            "summary": summarize(rows),
        }
    full = result["panels"]["hancock_mechanisms"]
    full_loci = full["loci"]
    full_rows = full["rows"]
    result["derived_sensitivities"] = {
        "per_locus": {
            locus: {
                "population_ratios": {
                    row["cohort"]: row["loci"][locus]["ratio"] for row in full_rows
                },
                "median_ratio": float(np.nanmedian([
                    row["loci"][locus]["ratio"] for row in full_rows
                ])),
            }
            for locus in full_loci
        },
        "leave_one_locus": {},
    }
    for omitted in full_loci:
        population_ratios = {}
        for row in full_rows:
            ratios = np.asarray([
                row["loci"][locus]["ratio"]
                for locus in full_loci if locus != omitted
            ], dtype=float)
            ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
            population_ratios[row["cohort"]] = (
                float(np.exp(np.mean(np.log(ratios)))) if len(ratios) else float("nan")
            )
        result["derived_sensitivities"]["leave_one_locus"][omitted] = {
            "population_ratios": population_ratios,
            "median_ratio": float(np.nanmedian(list(population_ratios.values()))),
        }
    no_overlap = result["derived_sensitivities"]["leave_one_locus"]["Cyp4j5"]
    result["derived_sensitivities"]["overlap_deduplicated_14_region"] = {
        **no_overlap,
        "omitted": "Cyp4j5",
        "reason": "its +/-0.15-Mb window partly overlaps the Rdl window",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}")
    for name, panel in result["panels"].items():
        summary = panel["summary"]
        print(name, summary["median_ratio"], summary["n_nominal_p_lt_0_05"], "/", summary["n_cohorts"])


if __name__ == "__main__":
    main()
