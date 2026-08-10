#!/usr/bin/env python3
"""Call Phase 2 crossovers from released SHAPEIT cross haplotypes.

The release does not contain genotype-quality and site-filter arrays for the
crossing families. This script therefore treats the released phased,
QC-filtered autosomal haplotypes as its input and carries that limitation into
the output.  Marker thinning, parental orientation, the two-state caller, and
the three-marker flanking rule follow the documented calling procedure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ARMS = ("2R", "2L", "3R", "3L")
ARM_LENGTH = {"2R": 61_545_105, "2L": 49_364_325, "3R": 53_200_684, "3L": 41_963_435}
BIN_BP = 10_000
WINDOW_BP = 5_000_000
MAX_WIDTH_BP = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    families = {}
    for cross in sorted({row["cross"] for row in rows}):
        group = [row for row in rows if row["cross"] == cross]
        parents = [row for row in group if row["role"] == "parent"]
        children = [row for row in group if row["role"] == "progeny"]
        mother = next(row["ox_code"] for row in parents if row["sex"] == "F")
        father = next(row["ox_code"] for row in parents if row["sex"] == "M")
        families[cross] = {
            "mother": mother,
            "father": father,
            "offspring": [row["ox_code"] for row in children],
        }
    return families


def sample_order(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        rows = [line.split()[0] for line in handle.readlines()[2:]]
    if len(rows) != len(set(rows)):
        raise ValueError(f"duplicate samples in {path}")
    return rows


def empty_marker(n_child: int) -> dict[str, list[np.ndarray]]:
    return {key: [] for key in (
        "position", "transmission", "transmission_status", "offspring_gq",
        "called_offspring", "mendelian_errors", "mendelian_error_fraction",
    )}


def append_markers(target: dict[str, list[np.ndarray]], block: dict[str, np.ndarray]) -> None:
    for key in target:
        target[key].append(block[key])


def concatenate(target: dict[str, list[np.ndarray]], n_child: int) -> dict[str, np.ndarray]:
    output = {}
    two_dimensional = {"transmission", "transmission_status", "offspring_gq"}
    for key, blocks in target.items():
        if blocks:
            output[key] = np.concatenate(blocks, axis=0)
        elif key in two_dimensional:
            dtype = np.int16 if key == "offspring_gq" else np.int8
            output[key] = np.empty((0, n_child), dtype=dtype)
        else:
            dtype = np.float32 if key == "mendelian_error_fraction" else np.int64
            output[key] = np.empty(0, dtype=dtype)
    return output


def extract_arm(haps: Path, samples_file: Path, families: dict, core) -> dict:
    samples = sample_order(samples_file)
    sample_index = {sample: index for index, sample in enumerate(samples)}
    missing = sorted({sample for family in families.values() for sample in [
        family["mother"], family["father"], *family["offspring"]
    ] if sample not in sample_index})
    if missing:
        raise ValueError(f"cross samples absent from {samples_file}: {missing}")
    collected = {
        cross: {
            "mother": empty_marker(len(family["offspring"])),
            "father": empty_marker(len(family["offspring"])),
        }
        for cross, family in families.items()
    }
    chunk_position, chunk_genotype = [], []

    def process_chunk() -> None:
        if not chunk_position:
            return
        position = np.asarray(chunk_position, dtype=np.int64)
        genotype = np.asarray(chunk_genotype, dtype=np.int8).reshape(len(position), len(samples), 2)
        alleles = np.zeros((len(position), 3), dtype=np.int8)
        passed = np.ones(len(position), dtype=bool)
        gq = np.full((len(position), len(samples)), 99, dtype=np.int16)
        filters = core.MarkerFilters(
            parent_gq_min=0,
            offspring_gq_min=0,
            minimum_called_offspring=8,
            maximum_mendelian_error_fraction=0.05,
        )
        for cross, family in families.items():
            offspring = [sample_index[sample] for sample in family["offspring"]]
            mother = sample_index[family["mother"]]
            father = sample_index[family["father"]]
            for parent, target, mate in (("mother", mother, father), ("father", father, mother)):
                marker = core.informative_transmissions(
                    positions=position,
                    alleles=alleles,
                    filter_pass=passed,
                    genotypes=genotype,
                    genotype_quality=gq,
                    target_index=target,
                    mate_index=mate,
                    offspring_indices=offspring,
                    filters=filters,
                )
                append_markers(collected[cross][parent], marker)

    with gzip.open(haps, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 5 + 2 * len(samples):
                raise ValueError(f"unexpected column count at {haps}:{line_number}")
            chunk_position.append(int(fields[2]))
            chunk_genotype.append(np.fromiter(
                (-1 if int(value) > 1 else int(value) for value in fields[5:]),
                dtype=np.int8,
            ))
            if len(chunk_position) == 20_000:
                process_chunk()
                chunk_position.clear()
                chunk_genotype.clear()
        process_chunk()
    return {
        cross: {
            parent: concatenate(values, len(families[cross]["offspring"]))
            for parent, values in parents.items()
        }
        for cross, parents in collected.items()
    }


def overlap_bp(left: int, right: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.minimum(right, ends) - np.maximum(left, starts))


def call_parent(core, raw: dict, n_child: int, arm: str) -> dict:
    marker = core.thin_transmission_markers(raw, bin_bp=BIN_BP)
    phased = core.phase_transmission_markers(
        marker["transmission"],
        minimum_overlap=8,
        minimum_agreement=0.70,
        maximum_discordant_offspring=1,
    )
    events = core.call_crossovers(
        marker["position"],
        phased["oriented"],
        phased["phase_block"],
        genotype_error_probability=0.0025,
        rate_per_bp=1e-8,
        minimum_state_run_markers=3,
    )
    starts = np.arange(0, ARM_LENGTH[arm], WINDOW_BP, dtype=np.int64)
    ends = np.minimum(starts + WINDOW_BP, ARM_LENGTH[arm])
    detection = np.zeros(len(starts), float)
    marker_span = np.zeros(len(starts), float)
    for child in range(n_child):
        for block in np.unique(phased["phase_block"]):
            index = np.flatnonzero(
                (phased["phase_block"] == block) & (phased["oriented"][:, child] >= 0)
            )
            if len(index) < 2:
                continue
            positions = marker["position"][index]
            marker_span += overlap_bp(int(positions[0]), int(positions[-1]), starts, ends)
            if len(positions) >= 6:
                detection += overlap_bp(int(positions[2]), int(positions[-3]), starts, ends)
    retained = events[events["width_bp"] <= MAX_WIDTH_BP]
    counts = np.histogram(retained["midpoint_bp"], bins=np.r_[starts, ends[-1]])[0].astype(float)
    return {
        "raw_markers": len(raw["position"]),
        "thinned_markers": len(marker["position"]),
        "phase_blocks": len(np.unique(phased["phase_block"])),
        "events": retained,
        "event_count": counts,
        "detection_exposure_bp": detection,
        "marker_span_bp": marker_span,
        "starts": starts,
        "ends": ends,
    }


def atlas_consensus(maps: Path, cohorts: list[str], arm: str) -> np.ndarray:
    rows = []
    for cohort in cohorts:
        with np.load(maps / f"{cohort}__{arm}.npz", allow_pickle=True) as z:
            starts_50 = z["starts_50000"].astype(float)
            rates_50 = z["r_50000"].astype(float)
        starts = np.arange(0, ARM_LENGTH[arm], WINDOW_BP, dtype=np.int64)
        values = []
        for start in starts:
            keep = (starts_50 >= start) & (starts_50 < start + WINDOW_BP) & np.isfinite(rates_50)
            values.append(float(np.mean(rates_50[keep])) if np.any(keep) else np.nan)
        values = np.asarray(values)
        values /= np.nanmean(values)
        rows.append(values)
    return np.nanmedian(np.asarray(rows), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapeit", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument("--calls-only", action="store_true")
    stage.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.core_dir))
    import core  # type: ignore

    families = read_metadata(args.metadata)
    with args.selection.open(newline="", encoding="utf-8") as handle:
        cohorts = [row["cohort"] for row in csv.DictReader(handle, delimiter="\t")]
    aggregate = defaultdict(lambda: defaultdict(float))
    call_rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        calls = None
        if not args.score_only:
            calls = extract_arm(
                args.shapeit / f"ag1000g.phase2.ar1.haplotypes.{arm}.gz",
                args.shapeit / f"ag1000g.phase2.ar1.samples.{arm}.gz",
                families,
                core,
            )
        for cross, family in families.items():
            for parent in ("mother", "father"):
                event_path = args.out / "calls" / cross / arm / f"{parent}.npz"
                if args.score_only:
                    if not event_path.is_file():
                        raise FileNotFoundError(event_path)
                    with np.load(event_path, allow_pickle=False) as archive:
                        called = {key: archive[key] for key in archive.files}
                else:
                    called = call_parent(core, calls[cross][parent], len(family["offspring"]), arm)
                    event_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(event_path, **called)
                for key in ("event_count", "detection_exposure_bp", "marker_span_bp"):
                    aggregate[arm][key] += called[key]
                call_rows.append({
                    "cross": cross,
                    "arm": arm,
                    "parent": parent,
                    "offspring": len(family["offspring"]),
                    "raw_markers": int(called["raw_markers"]),
                    "thinned_markers": int(called["thinned_markers"]),
                    "phase_blocks": int(called["phase_blocks"]),
                    "events_width_le_1mb": len(called["events"]),
                    "median_breakpoint_width_bp": float(np.median(called["events"]["width_bp"])) if len(called["events"]) else None,
                    "file": str(event_path.relative_to(args.out)),
                    "sha256": sha256(event_path),
                })
        print(f"finished {arm}", flush=True)

    if args.calls_only:
        call_manifest = {
            "schema_version": 1,
            "variant": "phase2",
            "release": "Ag1000G Phase 2 AR1",
            "n_call_files": len(call_rows),
            "calls": call_rows,
            "provenance": {
                "cross_metadata_sha256": sha256(args.metadata),
                "core_sha256": sha256(args.core_dir / "core.py"),
                "shapeit_sha256": {
                    arm: sha256(args.shapeit / f"ag1000g.phase2.ar1.haplotypes.{arm}.gz") for arm in ARMS
                },
            },
        }
        (args.out / "phase2_pedigree_call_manifest.json").write_text(
            json.dumps(call_manifest, indent=2) + "\n"
        )
        print(f"wrote {args.out / 'phase2_pedigree_call_manifest.json'}")
        return

    direct, atlas, arm_label, support = [], [], [], []
    arm_data = {}
    windows = []
    for arm in ARMS:
        starts = np.arange(0, ARM_LENGTH[arm], WINDOW_BP, dtype=np.int64)
        ends = np.minimum(starts + WINDOW_BP, ARM_LENGTH[arm])
        nominal = sum(2 * len(family["offspring"]) for family in families.values()) * (ends - starts)
        supported = aggregate[arm]["marker_span_bp"] >= 0.8 * nominal
        exposure = aggregate[arm]["detection_exposure_bp"]
        rate = np.divide(
            aggregate[arm]["event_count"], exposure,
            out=np.full(len(starts), np.nan), where=exposure > 0,
        ) * 1e8
        rate /= np.nanmean(rate[supported])
        consensus = atlas_consensus(args.maps, cohorts, arm)
        for index in range(len(starts)):
            windows.append({
                "arm": arm, "start": int(starts[index]), "end": int(ends[index]),
                "events": float(aggregate[arm]["event_count"][index]),
                "exposure_bp": float(exposure[index]),
                "marker_span_fraction": float(aggregate[arm]["marker_span_bp"][index] / nominal[index]),
                "direct_normalized": float(rate[index]) if np.isfinite(rate[index]) else None,
                "atlas_normalized": float(consensus[index]) if np.isfinite(consensus[index]) else None,
                "supported": bool(supported[index]),
            })
        keep = supported & np.isfinite(rate) & np.isfinite(consensus)
        arm_data[arm] = (rate, consensus, keep)
        direct.extend(rate[keep])
        atlas.extend(consensus[keep])
        arm_label.extend([arm] * int(np.sum(keep)))
        support.extend(np.flatnonzero(keep).tolist())
    statistic = spearmanr(direct, atlas)
    null = []
    shift_ranges = [range(len(arm_data[arm][1])) for arm in ARMS]
    for shifts in itertools.product(*shift_ranges):
        if not any(shifts):
            continue
        shifted = []
        for arm, shift in zip(ARMS, shifts):
            _rate, consensus, keep = arm_data[arm]
            shifted.extend(np.roll(consensus, shift)[keep])
        null.append(float(spearmanr(direct, shifted).statistic))
    null_array = np.asarray(null, dtype=float)
    spatial_p = float(np.mean(np.abs(null_array) >= abs(float(statistic.statistic))))
    result = {
        "schema_version": 1,
        "variant": "phase2",
        "release": "Ag1000G Phase 2 AR1",
        "input_limitation": (
            "Calls use released SHAPEIT haplotypes without genotype-quality and site-filter arrays; "
            "the result is a release-specific broad-scale comparison."
        ),
        "n_crosses": len(families),
        "crosses": sorted(families),
        "n_progeny": sum(len(family["offspring"]) for family in families.values()),
        "n_parental_transmissions_per_arm": sum(2 * len(family["offspring"]) for family in families.values()),
        "n_events_width_le_1mb": sum(row["events_width_le_1mb"] for row in call_rows),
        "n_supported_5mb_windows": len(direct),
        "spearman_5mb": [float(statistic.statistic), spatial_p],
        "spearman_5mb_asymptotic_p": float(statistic.pvalue),
        "spatial_shift_null": "all nonzero combinations of whole-window circular shifts within arms",
        "n_spatial_shifts": len(null),
        "calls": call_rows,
        "windows": windows,
        "provenance": {
            "cross_metadata_sha256": sha256(args.metadata),
            "core_sha256": sha256(args.core_dir / "core.py"),
            "shapeit_sha256": {
                arm: sha256(args.shapeit / f"ag1000g.phase2.ar1.haplotypes.{arm}.gz") for arm in ARMS
            },
        },
    }
    (args.out / "phase2_pedigree.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.out / "phase2_pedigree_windows.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=windows[0].keys())
        writer.writeheader()
        writer.writerows(windows)
    print(json.dumps({key: result[key] for key in ("n_crosses", "n_events_width_le_1mb", "n_supported_5mb_windows", "spearman_5mb")}, indent=2))


if __name__ == "__main__":
    main()
