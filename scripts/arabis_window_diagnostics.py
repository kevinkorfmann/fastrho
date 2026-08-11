"""Cross-map-blind diagnostics for Arabis empirical-map windows.

The diagnostic deliberately never reads the F2 rate arrays.  It asks whether
features of the population-WGS input, chunk geometry, or model-to-model
variation can explain conspicuous inferred-map windows.  All thresholds are
generic 3-IQR outer fences computed genome-wide within each species.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from arabis_cross_eval import binned_prediction, relative
from arabis_infer import read_selfer_vcf


CHROMS = tuple(f"chr{i}" for i in range(1, 9))
SPECIES = ("nemorensis", "sagittata")
VCFS = {
    "nemorensis": "nemorensis.selfer.vcf.gz",
    "sagittata": "sagittata.selfer.vcf.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap_weighted_mean(
    left: np.ndarray,
    right: np.ndarray,
    values: np.ndarray,
    lo: float,
    hi: float,
) -> float:
    overlap = np.maximum(0.0, np.minimum(right, hi) - np.maximum(left, lo))
    return float(np.sum(overlap * values) / np.sum(overlap))


def chunk_membership(n_sites: int, context_len: int, overlap: int) -> np.ndarray:
    """Number of inference chunks contributing to each SNP interval."""
    stride = max(1, context_len - overlap)
    starts = list(range(0, max(1, n_sites - context_len + 1), stride))
    if starts[-1] != max(0, n_sites - context_len):
        starts.append(max(0, n_sites - context_len))
    membership = np.zeros(n_sites - 1, dtype=np.int16)
    for start in starts:
        stop = min(n_sites, start + context_len)
        membership[start : max(start, stop - 1)] += 1
    if np.any(membership < 1):
        raise RuntimeError("chunk geometry left an interval uncovered")
    return membership


def adjacent_r2(gm: np.ndarray) -> np.ndarray:
    x = gm[:, :-1].astype(float)
    y = gm[:, 1:].astype(float)
    px, py = np.mean(x, axis=0), np.mean(y, axis=0)
    covariance = np.mean(x * y, axis=0) - px * py
    denominator = px * (1.0 - px) * py * (1.0 - py)
    return np.divide(
        covariance**2,
        denominator,
        out=np.full_like(covariance, np.nan),
        where=denominator > 0,
    )


def outer_fences(values: np.ndarray) -> tuple[float, float]:
    q1, q3 = np.quantile(values, (0.25, 0.75))
    iqr = q3 - q1
    return float(q1 - 3.0 * iqr), float(q3 + 3.0 * iqr)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if np.sum(keep) < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=int, default=7)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    args = parser.parse_args()

    # Only the physical edges are loaded.  The F2 arrays in this archive are
    # intentionally neither indexed nor included in any decision.
    grids = np.load(args.windows_npz, allow_pickle=False)
    maps_root = args.campaign_root / "arabis_maps"
    rows: list[dict] = []
    for species in SPECIES:
        vcf = args.workdir / "vcf" / VCFS[species]
        for chrom in CHROMS:
            gm, positions = read_selfer_vcf(vcf, chrom)
            edges = np.asarray(grids[f"{chrom}_edges"], float)
            ensemble_path = maps_root / "ensemble" / f"{species}.{chrom}.npz"
            ensemble_rate = relative(binned_prediction(ensemble_path, edges))
            seed_rates = np.stack(
                [
                    relative(
                        binned_prediction(
                            maps_root / f"seed{seed}" / f"{species}.{chrom}.npz", edges
                        )
                    )
                    for seed in range(args.members)
                ]
            )
            with np.load(ensemble_path, allow_pickle=False) as prediction:
                left = np.asarray(prediction["pos_left"], float)
                right = np.asarray(prediction["pos_right"], float)
                sigma = np.asarray(prediction["sigma_log_rho"], float)
            membership = chunk_membership(len(positions), args.context_len, args.overlap)
            r2 = adjacent_r2(gm)
            midpoint = (positions[:-1] + positions[1:]) / 2.0
            mac = np.minimum(np.sum(gm, axis=0), gm.shape[0] - np.sum(gm, axis=0))

            for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                site_mask = (positions >= lo) & (positions < hi)
                pair_mask = (midpoint >= lo) & (midpoint < hi)
                interval_mask = (right > lo) & (left < hi)
                full_gaps = right[interval_mask] - left[interval_mask]
                one_chunk = (membership == 1).astype(float)
                row = {
                    "species": species,
                    "chromosome": chrom,
                    "window_index": index,
                    "start_bp": int(lo),
                    "end_bp": int(hi),
                    "center_mb": float((lo + hi) / 2e6),
                    "structured_relative_rate": float(ensemble_rate[index]),
                    "seed_relative_rate_mean": float(np.mean(seed_rates[:, index])),
                    "seed_relative_rate_sd": float(np.std(seed_rates[:, index], ddof=1)),
                    "seed_relative_rate_min": float(np.min(seed_rates[:, index])),
                    "seed_relative_rate_max": float(np.max(seed_rates[:, index])),
                    "seed_relative_rates": [
                        float(value) for value in seed_rates[:, index]
                    ],
                    "n_snps": int(np.sum(site_mask)),
                    "median_mac": float(np.median(mac[site_mask])),
                    "rare_variant_fraction": float(np.mean(mac[site_mask] <= 2)),
                    "adjacent_r2_mean": float(np.nanmean(r2[pair_mask])),
                    "max_inter_snp_gap_bp": int(np.max(full_gaps)),
                    "median_inter_snp_gap_bp": float(np.median(full_gaps)),
                    "single_chunk_span_fraction": overlap_weighted_mean(
                        left, right, one_chunk, lo, hi
                    ),
                    "mean_chunk_membership": overlap_weighted_mean(
                        left, right, membership.astype(float), lo, hi
                    ),
                    "mean_interval_sigma_log_rho": overlap_weighted_mean(
                        left, right, sigma, lo, hi
                    ),
                    "crosses_5mb_call_boundary": bool(
                        any(lo < boundary < hi for boundary in range(5_000_000, 60_000_000, 5_000_000))
                    ),
                }
                rows.append(row)

    summaries = {}
    for species in SPECIES:
        selected = [row for row in rows if row["species"] == species]
        rate = np.asarray([row["structured_relative_rate"] for row in selected])
        diagnostics = {
            key: np.asarray([row[key] for row in selected], float)
            for key in (
                "n_snps",
                "rare_variant_fraction",
                "adjacent_r2_mean",
                "max_inter_snp_gap_bp",
                "single_chunk_span_fraction",
                "seed_relative_rate_sd",
                "mean_interval_sigma_log_rho",
            )
        }
        fences = {key: outer_fences(value[np.isfinite(value)]) for key, value in diagnostics.items()}
        for row in selected:
            row["low_snp_support_extreme"] = row["n_snps"] < fences["n_snps"][0]
            row["large_gap_extreme"] = row["max_inter_snp_gap_bp"] > fences["max_inter_snp_gap_bp"][1]
            row["seed_instability_extreme"] = row["seed_relative_rate_sd"] > fences["seed_relative_rate_sd"][1]
            row["posterior_uncertainty_extreme"] = row["mean_interval_sigma_log_rho"] > fences["mean_interval_sigma_log_rho"][1]
            row["technical_support_flag"] = bool(
                row["low_snp_support_extreme"]
                or row["large_gap_extreme"]
                or row["seed_instability_extreme"]
                or row["posterior_uncertainty_extreme"]
            )
        boundary = np.asarray([row["crosses_5mb_call_boundary"] for row in selected], bool)
        summaries[species] = {
            "n_windows": len(selected),
            "outer_fences": {key: list(value) for key, value in fences.items()},
            "rate_spearman_with_diagnostics": {
                key: safe_spearman(rate, value) for key, value in diagnostics.items()
            },
            "mean_rate_crossing_5mb_call_boundary": float(np.mean(rate[boundary])),
            "mean_rate_not_crossing_5mb_call_boundary": float(np.mean(rate[~boundary])),
            "technical_support_flag_count": int(
                sum(row["technical_support_flag"] for row in selected)
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "design": "cross-map-blind population-WGS support and model-stability audit",
        "f2_rate_arrays_read": False,
        "decision_rule": "generic species-wise 3-IQR outer fences; no threshold uses the F2 map",
        "context_len_snps": args.context_len,
        "overlap_snps": args.overlap,
        "ensemble_members": args.members,
        "windows_npz_sha256": sha256(args.windows_npz),
        "summaries": summaries,
        "windows": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
