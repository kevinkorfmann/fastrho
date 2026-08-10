"""Benchmark historical Arabis LD maps against the Rahnamae et al. F2 map.

The primary comparison concerns chromosome-relative shape at 2 Mb, not absolute
rate: the F2 map measures contemporary recombination in one interspecific cross,
whereas fastrho reads the longer historical record within each parental species.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

CHROMS = tuple(f"chr{i}" for i in range(1, 9))
SPECIES = ("nemorensis", "sagittata")
SOURCE_COMMIT = "10c9092ce9e08c16b0a958d4062932262a8c6bc4"
MODEL_CAMPAIGN = "campaign_self2"
MODEL_TRAINING_N_LINES = [50, 80, 120, 156, 200]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_sheet_audit(path: Path) -> dict:
    accessions = {species: [] for species in SPECIES}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accessions[row["species"]].append(row["accession"])
    return {
        "sha256": sha256(path),
        "counts": {species: len(accessions[species]) for species in SPECIES},
        "accessions": {species: sorted(accessions[species], key=int) for species in SPECIES},
    }


def read_versions(path: Path) -> dict[str, str]:
    versions = {}
    with path.open() as handle:
        for line in handle:
            name, version = line.rstrip().split("\t", 1)
            versions[name] = version
    return versions


def pava(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted pool-adjacent-violators fit for a nondecreasing series."""
    y = np.asarray(values, float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, float)
    level, weight, start, end = [], [], [], []
    for i, (yi, wi) in enumerate(zip(y, w)):
        level.append(float(yi))
        weight.append(float(wi))
        start.append(i)
        end.append(i)
        while len(level) > 1 and level[-2] > level[-1]:
            nw = weight[-2] + weight[-1]
            level[-2] = (level[-2] * weight[-2] + level[-1] * weight[-1]) / nw
            weight[-2] = nw
            end[-2] = end[-1]
            level.pop()
            weight.pop()
            start.pop()
            end.pop()
    out = np.empty_like(y)
    for value, lo, hi in zip(level, start, end):
        out[lo : hi + 1] = value
    return out


def read_cross_map(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    raw = {chrom: [] for chrom in CHROMS}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for marker, chrom, cm, *_ in reader:
            name = f"chr{int(chrom)}"
            raw[name].append((float(marker.rsplit("_", 1)[1]) * 1e6, float(cm)))
    fitted, audit = {}, {"markers": 0, "chromosomes": {}}
    for chrom, records in raw.items():
        by_position: dict[float, list[float]] = {}
        for bp, cm in records:
            by_position.setdefault(bp, []).append(cm)
        bp = np.asarray(sorted(by_position), float)
        cm = np.asarray([np.median(by_position[x]) for x in bp], float)
        orientation = 1 if spearmanr(bp, cm).statistic >= 0 else -1
        if orientation < 0:
            cm = np.max(cm) - cm
        monotone = pava(cm)
        fitted[chrom] = (bp, monotone)
        audit["markers"] += len(records)
        audit["chromosomes"][chrom] = {
            "raw_markers": len(records),
            "unique_positions": len(bp),
            "orientation": orientation,
            "isotonic_rms_cm": float(np.sqrt(np.mean((monotone - cm) ** 2))),
            "physical_span_mb": float((bp[-1] - bp[0]) / 1e6),
            "map_span_cm": float(monotone[-1] - monotone[0]),
        }
    return fitted, audit


def integral_at(x: np.ndarray, left: np.ndarray, right: np.ndarray, rate: np.ndarray) -> np.ndarray:
    widths = right - left
    cumulative = np.r_[0.0, np.cumsum(widths * rate)]
    idx = np.searchsorted(right, x, side="right")
    idx = np.clip(idx, 0, len(rate) - 1)
    return cumulative[idx] + np.clip(x - left[idx], 0, widths[idx]) * rate[idx]


def binned_prediction(npz: Path, edges: np.ndarray) -> np.ndarray:
    with np.load(npz) as z:
        left, right, rate = (
            np.asarray(z[k], float) for k in ("pos_left", "pos_right", "rho_per_bp")
        )
    if edges[0] < left[0] or edges[-1] > right[-1]:
        raise ValueError(f"prediction does not span requested grid: {npz}")
    return np.diff(integral_at(edges, left, right, rate)) / np.diff(edges)


def prediction_span(npz: Path) -> tuple[float, float]:
    with np.load(npz) as z:
        return float(np.asarray(z["pos_left"])[0]), float(np.asarray(z["pos_right"])[-1])


def relative(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("map has no positive finite mean")
    return values / mean


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def chromosome_bootstrap(
    by_chrom: dict[str, tuple[np.ndarray, np.ndarray]], seed: int, draws: int
) -> list[float]:
    rng, chroms, out = np.random.default_rng(seed), sorted(by_chrom), []
    for _ in range(draws):
        selected = rng.choice(chroms, len(chroms), replace=True)
        x = np.concatenate([by_chrom[c][0] for c in selected])
        y = np.concatenate([by_chrom[c][1] for c in selected])
        out.append(float(spearmanr(x, y).statistic))
    return out


def circular_null(
    by_chrom: dict[str, tuple[np.ndarray, np.ndarray]], seed: int, draws: int
) -> list[float]:
    rng, out = np.random.default_rng(seed), []
    for _ in range(draws):
        x, y = [], []
        for chrom in sorted(by_chrom):
            a, b = by_chrom[chrom]
            shift = int(rng.integers(1, max(2, len(b)))) if len(b) > 1 else 0
            x.append(a)
            y.append(np.roll(b, shift))
        out.append(float(spearmanr(np.concatenate(x), np.concatenate(y)).statistic))
    return out


def evaluate_resolution(
    cross,
    maps_dir: Path,
    width: int,
    draws: int,
    map_stems: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    map_stems = map_stems or {species: species for species in SPECIES}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    paired = {species: {} for species in (*SPECIES, "consensus")}
    per_chrom = {}
    for chrom in CHROMS:
        bp, cm = cross[chrom]
        spans = [prediction_span(maps_dir / f"{map_stems[s]}.{chrom}.npz") for s in SPECIES]
        start = np.ceil(max(bp[0], *(span[0] for span in spans)) / width) * width
        stop = np.floor(min(bp[-1], *(span[1] for span in spans)) / width) * width
        if stop - start < 2 * width:
            raise ValueError(f"insufficient common span for {chrom} at {width / 1e6:g} Mb")
        edges = np.arange(start, stop + width, width, dtype=float)
        cross_rate = relative(np.diff(np.interp(edges, bp, cm)) / (width / 1e6))
        pred = {
            s: relative(binned_prediction(maps_dir / f"{map_stems[s]}.{chrom}.npz", edges))
            for s in SPECIES
        }
        pred["consensus"] = relative(np.sqrt(pred["nemorensis"] * pred["sagittata"]))
        arrays[chrom] = {"edges": edges, "cross": cross_rate, **pred}
        per_chrom[chrom] = {s: correlations(cross_rate, pred[s]) for s in pred}
        for s in pred:
            paired[s][chrom] = (cross_rate, pred[s])

    summary = {"window_mb": width / 1e6, "per_chromosome": per_chrom, "maps": {}}
    topology = {}
    for key in ("cross", *SPECIES, "consensus"):
        outer, center = [], []
        for chrom in CHROMS:
            values = arrays[chrom][key]
            edges = arrays[chrom]["edges"]
            midpoint = (edges[:-1] + edges[1:]) / 2
            fraction = (midpoint - edges[0]) / (edges[-1] - edges[0])
            is_outer = (fraction < 0.25) | (fraction > 0.75)
            outer.append(values[is_outer])
            center.append(values[~is_outer])
        topology[key] = float(np.mean(np.concatenate(outer)) / np.mean(np.concatenate(center)))
    summary["outer_quarters_to_center_half_ratio"] = topology
    for i, species in enumerate((*SPECIES, "consensus")):
        x = np.concatenate([paired[species][c][0] for c in CHROMS])
        y = np.concatenate([paired[species][c][1] for c in CHROMS])
        boot = chromosome_bootstrap(paired[species], 70779 + i + width, draws)
        null = circular_null(paired[species], 170779 + i + width, draws)
        distorted = [c for c in CHROMS if c not in {"chr4", "chr7"}]
        xd = np.concatenate([paired[species][c][0] for c in distorted])
        yd = np.concatenate([paired[species][c][1] for c in distorted])
        loco = {}
        for left_out in CHROMS:
            kept = [c for c in CHROMS if c != left_out]
            loco[left_out] = float(
                spearmanr(
                    np.concatenate([paired[species][c][0] for c in kept]),
                    np.concatenate([paired[species][c][1] for c in kept]),
                ).statistic
            )
        obs = correlations(x, y)
        summary["maps"][species] = {
            **obs,
            "n_windows": len(x),
            "spearman_chromosome_bootstrap_ci95": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "circular_shift_p_one_sided": float(
                (1 + np.sum(np.asarray(null) >= obs["spearman"])) / (draws + 1)
            ),
            "spearman_excluding_distorted_chr4_chr7": float(spearmanr(xd, yd).statistic),
            "spearman_leave_one_chromosome_out": loco,
        }
    a = np.concatenate([arrays[c]["nemorensis"] for c in CHROMS])
    b = np.concatenate([arrays[c]["sagittata"] for c in CHROMS])
    summary["between_species"] = correlations(a, b)
    return summary, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-map", type=Path, required=True)
    parser.add_argument("--sample-sheet", type=Path, required=True)
    parser.add_argument("--software-versions", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--model-campaign", default=MODEL_CAMPAIGN)
    parser.add_argument("--model-training-n-lines", default=None,
                        help="comma-separated simulated panel sizes")
    args = parser.parse_args()

    cross, audit = read_cross_map(args.cross_map)
    input_audit = {}
    for species in SPECIES:
        chromosomes, total_snps = {}, 0
        for chrom in CHROMS:
            with np.load(args.maps_dir / f"{species}.{chrom}.npz") as z:
                row = {
                    "n_accessions": int(z["n_accessions"]),
                    "n_snps": int(z["n_snps"]),
                    "Ne_estimated": float(z["Ne_estimated"]),
                }
                chromosomes[chrom] = row
                total_snps += row["n_snps"]
                checkpoint_sha256 = str(z["checkpoint_sha256"])
                stats_sha256 = str(z["stats_sha256"])
        input_audit[species] = {
            "n_accessions": chromosomes["chr1"]["n_accessions"],
            "total_chromosome_snps": total_snps,
            "chromosomes": chromosomes,
        }
    results, primary_arrays = {}, None
    for width in (1_000_000, 2_000_000, 5_000_000):
        result, arrays = evaluate_resolution(cross, args.maps_dir, width, args.draws)
        results[f"{width // 1_000_000}Mb"] = result
        if width == 2_000_000:
            primary_arrays = arrays
    parent_stems = {species: f"{species}_no_cross_parent" for species in SPECIES}
    parent_result, parent_arrays = evaluate_resolution(
        cross, args.maps_dir, 2_000_000, args.draws, parent_stems
    )
    parent_input_audit = {}
    for species in SPECIES:
        chromosomes = {}
        for chrom in CHROMS:
            with np.load(args.maps_dir / f"{parent_stems[species]}.{chrom}.npz") as z:
                chromosomes[chrom] = {
                    "n_accessions": int(z["n_accessions"]),
                    "n_snps": int(z["n_snps"]),
                    "Ne_estimated": float(z["Ne_estimated"]),
                }
        parent_input_audit[species] = {
            "n_accessions": chromosomes["chr1"]["n_accessions"],
            "total_chromosome_snps": sum(row["n_snps"] for row in chromosomes.values()),
            "chromosomes": chromosomes,
        }
    payload = {
        "schema_version": 1,
        "primary_resolution": "2Mb",
        "interpretation": "chromosome-relative inferred recombination-map shape versus one contemporary interspecific F2 map",
        "source_commit": SOURCE_COMMIT,
        "model_campaign": args.model_campaign,
        "model_training_n_lines": (
            [int(value) for value in args.model_training_n_lines.split(",")]
            if args.model_training_n_lines else MODEL_TRAINING_N_LINES
        ),
        "cross_map_sha256": sha256(args.cross_map),
        "cross_map_audit": audit,
        "sample_assignments": sample_sheet_audit(args.sample_sheet),
        "variant_pipeline_software": read_versions(args.software_versions),
        "population_map_audit": input_audit,
        "parent_excluded_population_map_audit": parent_input_audit,
        "checkpoint_sha256": checkpoint_sha256,
        "stats_sha256": stats_sha256,
        "resolutions": results,
        "parent_excluded_2Mb": parent_result,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.npz.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    flat = {}
    assert primary_arrays is not None
    for chrom, values in primary_arrays.items():
        for key, value in values.items():
            flat[f"{chrom}_{key}"] = value
    for chrom, values in parent_arrays.items():
        for key, value in values.items():
            flat[f"parent_excluded_{chrom}_{key}"] = value
    np.savez_compressed(args.npz, **flat)


if __name__ == "__main__":
    main()
