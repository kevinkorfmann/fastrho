"""Frozen-checkpoint stress test for non-crossover gene conversion.

The experiment holds an exact variable crossover map fixed within each replicate and
adds spatially uniform gene-conversion initiation at several ratios to the mean
crossover rate.  The trained model is never refit.  Predictions are scored against
the crossover map at 25 kb, which asks whether gene conversion is misread as a change
in crossover shape or scale.

The JSON output contains the complete design and aggregate results.  The companion
NPZ retains every 25-kb truth and prediction array so all reported summaries can be
rederived without the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import msprime
import numpy as np
from scipy.stats import pearsonr, spearmanr

from fastrho.preprocess import mean_rate_between
from fastrho.simulate import RecombPriors, make_recombination_map
from fastrho.translate import load_model, predict_map_from_ts


def _parse_csv(values: str, cast):
    return tuple(cast(value.strip()) for value in values.split(",") if value.strip())


def _edges(sequence_length: int, window_size: int) -> np.ndarray:
    return np.append(np.arange(0, sequence_length, window_size, dtype=float), sequence_length)


def _prediction_on_grid(prediction: dict, edges: np.ndarray) -> np.ndarray:
    positions = np.concatenate([[prediction["pos_left"][0]], prediction["pos_right"]])
    rates = np.asarray(prediction["r_per_bp"], dtype=float)
    if positions[0] > edges[0]:
        positions = np.concatenate([[edges[0]], positions])
        rates = np.concatenate([[rates[0]], rates])
    if positions[-1] < edges[-1]:
        positions = np.concatenate([positions, [edges[-1]]])
        rates = np.concatenate([rates, [rates[-1]]])
    return mean_rate_between(positions, rates, edges)


def _metric(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(prediction) & np.isfinite(truth) & (prediction > 0) & (truth > 0)
    prediction = prediction[finite]
    truth = truth[finite]
    if prediction.size < 3:
        raise RuntimeError("fewer than three finite positive windows")
    return {
        "pearson": float(pearsonr(prediction, truth).statistic),
        "spearman": float(spearmanr(prediction, truth).statistic),
        "median_predicted_to_crossover": float(np.median(prediction / truth)),
        "mean_predicted_rate": float(np.mean(prediction)),
        "log_rmse": float(np.sqrt(np.mean((np.log(prediction) - np.log(truth)) ** 2))),
    }


def _pooled_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    return _metric(prediction.reshape(-1), truth.reshape(-1))


def _bootstrap_summary(
    prediction: np.ndarray,
    truth: np.ndarray,
    baseline: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, list[float] | float]:
    n_regions = prediction.shape[0]
    observed = _pooled_metrics(prediction, truth)
    observed["paired_mean_rate_vs_no_conversion"] = float(
        np.median(np.mean(prediction, axis=1) / np.mean(baseline, axis=1))
    )
    draws = {key: [] for key in observed}
    for _ in range(n_bootstrap):
        chosen = rng.integers(0, n_regions, n_regions)
        metrics = _pooled_metrics(prediction[chosen], truth[chosen])
        metrics["paired_mean_rate_vs_no_conversion"] = float(
            np.median(
                np.mean(prediction[chosen], axis=1) / np.mean(baseline[chosen], axis=1)
            )
        )
        for key, value in metrics.items():
            draws[key].append(value)
    return {
        key: {
            "estimate": float(observed[key]),
            "ci95": [float(x) for x in np.percentile(draws[key], [2.5, 97.5])],
        }
        for key in observed
    }


def run(args: argparse.Namespace) -> tuple[dict, dict[str, np.ndarray]]:
    ratios = _parse_csv(args.ratios, float)
    tract_lengths = _parse_csv(args.tract_lengths, int)
    if 0.0 not in ratios:
        ratios = (0.0,) + ratios
    conditions = [{"id": "gc0", "ratio": 0.0, "tract_length": 0}]
    conditions.extend(
        {
            "id": f"gc{ratio:g}_t{tract}",
            "ratio": float(ratio),
            "tract_length": int(tract),
        }
        for tract in tract_lengths
        for ratio in ratios
        if ratio > 0
    )

    model, config, stats = load_model(args.checkpoint, args.stats, device=args.device)
    edges = _edges(args.sequence_length, args.window_size)
    n_windows = edges.size - 1
    truth = np.full((args.n_regions, n_windows), np.nan, dtype=np.float64)
    predicted = np.full(
        (len(conditions), args.n_regions, n_windows), np.nan, dtype=np.float64
    )
    num_sites = np.zeros((len(conditions), args.n_regions), dtype=np.int64)
    region_map_kind: list[str] = []

    priors = RecombPriors(
        sequence_length=float(args.sequence_length),
        map_resolution=args.map_resolution,
    )
    for region_index in range(args.n_regions):
        map_kind = "gp" if region_index % 2 == 0 else "hotspot"
        region_map_kind.append(map_kind)
        map_rng = np.random.default_rng(args.seed + region_index)
        crossover_map = make_recombination_map(
            args.sequence_length,
            map_rng,
            kind=map_kind,
            mean_rate=args.crossover_rate,
            priors=priors,
        )
        mean_crossover = float(
            np.average(crossover_map.rate, weights=np.diff(crossover_map.position))
        )
        truth[region_index] = mean_rate_between(
            crossover_map.position, crossover_map.rate, edges
        )

        for condition_index, condition in enumerate(conditions):
            ancestry_kwargs = {
                "samples": args.n_diploid,
                "population_size": args.effective_size,
                "recombination_rate": crossover_map,
                "sequence_length": args.sequence_length,
                "random_seed": args.seed * 10000 + region_index * 100 + condition_index * 2 + 1,
            }
            if condition["ratio"] > 0:
                ancestry_kwargs["gene_conversion_rate"] = (
                    condition["ratio"] * mean_crossover
                )
                ancestry_kwargs["gene_conversion_tract_length"] = condition["tract_length"]
            tree_sequence = msprime.sim_ancestry(**ancestry_kwargs)
            tree_sequence = msprime.sim_mutations(
                tree_sequence,
                rate=args.mutation_rate,
                random_seed=args.seed * 10000 + region_index * 100 + condition_index * 2 + 2,
            )
            num_sites[condition_index, region_index] = tree_sequence.num_sites
            map_prediction = predict_map_from_ts(
                tree_sequence,
                model,
                config,
                stats,
                mutation_rate=args.mutation_rate,
                Ne=args.effective_size,
                device=args.device,
            )
            predicted[condition_index, region_index] = _prediction_on_grid(
                map_prediction, edges
            )
        print(
            f"completed region {region_index + 1}/{args.n_regions}",
            file=sys.stderr,
            flush=True,
        )

    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
    summaries = []
    baseline = predicted[0]
    for condition_index, condition in enumerate(conditions):
        region_metrics = [
            _metric(predicted[condition_index, region_index], truth[region_index])
            for region_index in range(args.n_regions)
        ]
        summaries.append(
            {
                **condition,
                "gene_conversion_rate": float(
                    condition["ratio"] * args.crossover_rate
                ),
                "median_num_sites": float(np.median(num_sites[condition_index])),
                "pooled": _bootstrap_summary(
                    predicted[condition_index],
                    truth,
                    baseline,
                    bootstrap_rng,
                    args.n_bootstrap,
                ),
                "region_median": {
                    key: float(np.median([row[key] for row in region_metrics]))
                    for key in region_metrics[0]
                },
            }
        )

    result = {
        "experiment": "gene_conversion_stress",
        "estimand": (
            "Recovery of the fixed crossover-rate map when gene conversion is present in "
            "the ancestry but absent from the checkpoint training model"
        ),
        "design": {
            "n_regions": args.n_regions,
            "sequence_length": args.sequence_length,
            "window_size": args.window_size,
            "n_diploid": args.n_diploid,
            "n_haplotypes": 2 * args.n_diploid,
            "effective_size": args.effective_size,
            "mutation_rate": args.mutation_rate,
            "mean_crossover_rate": args.crossover_rate,
            "gene_conversion_to_mean_crossover_ratios": list(ratios),
            "gene_conversion_tract_lengths": list(tract_lengths),
            "map_kinds": {"gp": args.n_regions // 2 + args.n_regions % 2,
                          "hotspot": args.n_regions // 2},
            "region_map_kind": region_map_kind,
            "paired_crossover_maps_across_conditions": True,
            "checkpoint_retrained": False,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "bootstrap_seed": args.bootstrap_seed,
            "msprime_version": msprime.__version__,
        },
        "conditions": summaries,
    }
    arrays = {
        "truth_25kb": truth,
        "predicted_25kb": predicted,
        "num_sites": num_sites,
        "condition_id": np.asarray([condition["id"] for condition in conditions]),
        "gene_conversion_ratio": np.asarray([condition["ratio"] for condition in conditions]),
        "tract_length": np.asarray([condition["tract_length"] for condition in conditions]),
        "window_edges": edges,
    }
    return result, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-regions", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=1_000_000)
    parser.add_argument("--window-size", type=int, default=25_000)
    parser.add_argument("--map-resolution", type=int, default=500)
    parser.add_argument("--n-diploid", type=int, default=10)
    parser.add_argument("--effective-size", type=float, default=10_000.0)
    parser.add_argument("--mutation-rate", type=float, default=1.5e-8)
    parser.add_argument("--crossover-rate", type=float, default=1e-8)
    parser.add_argument("--ratios", default="0,0.5,1,2,4")
    parser.add_argument("--tract-lengths", default="100,300,1000")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--bootstrap-seed", type=int, default=9302)
    args = parser.parse_args()

    result, arrays = run(args)
    out_json = Path(args.out_json)
    out_npz = Path(args.out_npz)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(out_npz, **arrays)
    print(out_json)
    print(out_npz)


if __name__ == "__main__":
    main()
