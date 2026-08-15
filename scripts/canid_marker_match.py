"""Prepare and summarize matched empirical dog/wolf sensitivity analyses.

The preparation stage makes 20 deterministic panel pairs.  Each replicate uses
33 village dogs (matching the 33 wolves), applies MAF >= 0.05, and retains the
same number of dog and wolf variants in every 100-kb x 0.05-MAF stratum.  The
resulting genotype archives are intended to be passed through the frozen canid
checkpoint with ``scripts/wolf_subset_infer.py``.

The summary stage combines the re-inferred maps at 100, 200, 500 and 1,000 kb.
Uncertainty jointly samples a deterministic panel replicate and 5-Mb genomic
blocks, so it reflects both ascertainment/subsampling and spatial dependence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


WINDOW = 100_000
SCALES = (100_000, 200_000, 500_000, 1_000_000)
# Lower edges for [0.05,0.10), ..., [0.45,0.50].  Do not add a separate
# 0.50 stratum: that would duplicate variants with MAF exactly one half.
MAF_EDGES = np.arange(0.05, 0.5, 0.05)


def _pair_rows(indices: np.ndarray) -> np.ndarray:
    return np.column_stack((2 * indices, 2 * indices + 1)).ravel()


def _maf(gm: np.ndarray) -> np.ndarray:
    af = np.asarray(gm, dtype=np.float64).mean(axis=0)
    return np.minimum(af, 1.0 - af)


def _save_subset(source, rows, sites, output: Path, key: str, sample_ids) -> None:
    payload = {name: source[name] for name in source.files
               if name not in {"gm", "pos", "pop", "n_ind", "sample_ids"}}
    payload.update(
        gm=np.asarray(source["gm"])[rows][:, sites].astype(np.int8, copy=False),
        pos=np.asarray(source["pos"])[sites].astype(np.int64, copy=False),
        pop=key,
        n_ind=len(rows) // 2,
        sample_ids=np.asarray(sample_ids),
        matching_window_bp=WINDOW,
        matching_maf_edges=MAF_EDGES,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def prepare(args) -> None:
    wolf = np.load(args.wolf, allow_pickle=True)
    dog = np.load(args.dog, allow_pickle=True)
    if wolf["gm"].shape[0] != 66:
        raise ValueError("the reference wolf panel must contain 33 diploid individuals")
    dog_n = dog["gm"].shape[0] // 2
    if dog_n < 33:
        raise ValueError("the dog panel must contain at least 33 diploid individuals")

    output = Path(args.output)
    hapdir = output / "hap"
    wolf_pos = np.asarray(wolf["pos"], dtype=np.int64)
    dog_pos = np.asarray(dog["pos"], dtype=np.int64)
    wolf_maf = _maf(wolf["gm"])
    wolf_ids = np.asarray(wolf.get("sample_ids", np.arange(33).astype(str)))
    dog_ids_all = np.asarray(dog.get("sample_ids", np.arange(dog_n).astype(str)))
    chrom_end = int(max(wolf_pos[-1], dog_pos[-1]))
    n_windows = int(np.ceil(chrom_end / WINDOW))
    manifest = {
        "description": (
            "Empirical canid marker-matched sensitivity: 33 dogs versus 33 wolves; "
            "exact variant-count matching within 100-kb windows and 0.05-wide MAF strata."
        ),
        "seed_base": args.seed,
        "replicates": args.replicates,
        "window_bp": WINDOW,
        "maf_min": 0.05,
        "maf_edges": MAF_EDGES.tolist(),
        "source_wolf": args.wolf,
        "source_dog": args.dog,
        "source_missing_policy": {
            "wolf": str(wolf.get("missing_policy", "unknown")),
            "dog": str(dog.get("missing_policy", "unknown")),
        },
        "replicate": [],
    }

    for replicate in range(args.replicates):
        seed = args.seed + replicate
        rng = np.random.default_rng(seed)
        dog_ind = np.sort(rng.choice(dog_n, 33, replace=False))
        dog_rows = _pair_rows(dog_ind)
        dog_gm = np.asarray(dog["gm"])[dog_rows]
        dog_maf = _maf(dog_gm)
        keep_wolf: list[np.ndarray] = []
        keep_dog: list[np.ndarray] = []
        per_window = []

        for window_index in range(n_windows):
            lo = window_index * WINDOW
            hi = lo + WINDOW
            wi = np.flatnonzero((wolf_pos >= lo) & (wolf_pos < hi))
            di = np.flatnonzero((dog_pos >= lo) & (dog_pos < hi))
            retained = 0
            strata = []
            for lower in MAF_EDGES:
                upper = lower + 0.05
                if upper >= 0.5 or np.isclose(upper, 0.5):
                    wm = wi[(wolf_maf[wi] >= lower) & (wolf_maf[wi] <= 0.5)]
                    dm = di[(dog_maf[di] >= lower) & (dog_maf[di] <= 0.5)]
                else:
                    wm = wi[(wolf_maf[wi] >= lower) & (wolf_maf[wi] < upper)]
                    dm = di[(dog_maf[di] >= lower) & (dog_maf[di] < upper)]
                n = min(len(wm), len(dm))
                if n:
                    keep_wolf.append(np.sort(rng.choice(wm, n, replace=False)))
                    keep_dog.append(np.sort(rng.choice(dm, n, replace=False)))
                retained += n
                strata.append(n)
            per_window.append({"window": window_index, "retained_each": retained,
                               "maf_strata_each": strata})

        wolf_sites = np.sort(np.concatenate(keep_wolf))
        dog_sites = np.sort(np.concatenate(keep_dog))
        if len(wolf_sites) != len(dog_sites):
            raise AssertionError("marker matching did not produce equal panel sizes")
        wolf_key = f"wolf_match_s{replicate:02d}"
        dog_key = f"dog_match_s{replicate:02d}"
        _save_subset(wolf, np.arange(66), wolf_sites, hapdir / f"{wolf_key}.npz",
                     wolf_key, wolf_ids)
        _save_subset(dog, dog_rows, dog_sites, hapdir / f"{dog_key}.npz",
                     dog_key, dog_ids_all[dog_ind])
        manifest["replicate"].append({
            "index": replicate,
            "seed": seed,
            "dog_individual_indices": dog_ind.tolist(),
            "dog_sample_ids": dog_ids_all[dog_ind].tolist(),
            "wolf_sample_ids": wolf_ids.tolist(),
            "variants_each": int(len(wolf_sites)),
            "windows_with_markers": int(sum(x["retained_each"] > 0 for x in per_window)),
            "per_window": per_window,
        })
        print(f"prepared replicate {replicate:02d}: {len(wolf_sites):,} variants per panel")

    with open(output / "matching_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def _aggregate(values: np.ndarray, factor: int) -> np.ndarray:
    n = len(values) // factor * factor
    return np.asarray(values[:n], dtype=float).reshape(-1, factor).mean(axis=1)


def _metric(pred: np.ndarray, truth: np.ndarray, name: str) -> float:
    valid = np.isfinite(pred) & np.isfinite(truth) & (pred > 0) & (truth > 0)
    x, y = pred[valid], truth[valid]
    if len(x) < 4:
        return float("nan")
    if name == "pearson":
        return float(pearsonr(x, y).statistic)
    if name == "log_pearson":
        return float(pearsonr(np.log(x), np.log(y)).statistic)
    if name == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(name)


def _block_draw(wolf, dog, factor, metric, rng, block_windows=50):
    n = min(len(wolf["pred"]), len(dog["pred"]))
    starts = np.arange(0, n, block_windows)
    chosen = rng.choice(starts, len(starts), replace=True)
    wp, wt, dp, dt = [], [], [], []
    for start in chosen:
        stop = min(start + block_windows, n)
        wp.append(_aggregate(wolf["pred"][start:stop], factor))
        wt.append(_aggregate(wolf["truth"][start:stop], factor))
        dp.append(_aggregate(dog["pred"][start:stop], factor))
        dt.append(_aggregate(dog["truth"][start:stop], factor))
    return (_metric(np.concatenate(wp), np.concatenate(wt), metric)
            - _metric(np.concatenate(dp), np.concatenate(dt), metric))


def summarize(args) -> None:
    root = Path(args.root)
    maps = root / "maps"
    pairs = []
    long_rows = []
    for replicate in range(args.replicates):
        wolf_path = maps / f"wolf_match_s{replicate:02d}.npz"
        dog_path = maps / f"dog_match_s{replicate:02d}.npz"
        if not wolf_path.exists() or not dog_path.exists():
            raise FileNotFoundError(f"missing inferred maps for replicate {replicate:02d}")
        wolf = np.load(wolf_path)
        dog = np.load(dog_path)
        pairs.append((wolf, dog))
        for scale in SCALES:
            factor = scale // WINDOW
            for species, archive in (("wolf", wolf), ("dog", dog)):
                pred = _aggregate(archive["pred"], factor)
                truth = _aggregate(archive["truth"], factor)
                for metric in ("pearson", "log_pearson", "spearman"):
                    long_rows.append({
                        "replicate": replicate, "species": species,
                        "window_bp": scale, "metric": metric,
                        "value": _metric(pred, truth, metric),
                    })

    output = {"description": (
        "Matched sample-size, local marker-density and MAF sensitivity. Confidence intervals "
        "jointly resample one of 20 deterministic matched panels and paired 5-Mb genomic blocks."
    ), "replicates": args.replicates, "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed, "block_size_bp": 5_000_000, "scales": {}}
    rng = np.random.default_rng(args.seed)
    for scale in SCALES:
        factor = scale // WINDOW
        scale_out = {}
        for metric in ("pearson", "log_pearson", "spearman"):
            wolf_values = np.array([r["value"] for r in long_rows
                                    if r["window_bp"] == scale and r["metric"] == metric
                                    and r["species"] == "wolf"])
            dog_values = np.array([r["value"] for r in long_rows
                                   if r["window_bp"] == scale and r["metric"] == metric
                                   and r["species"] == "dog"])
            draws = np.empty(args.bootstrap)
            for index in range(args.bootstrap):
                pair = pairs[int(rng.integers(0, len(pairs)))]
                draws[index] = _block_draw(pair[0], pair[1], factor, metric, rng)
            scale_out[metric] = {
                "wolf_median": float(np.nanmedian(wolf_values)),
                "dog_median": float(np.nanmedian(dog_values)),
                "wolf_minus_dog_median_across_replicates": float(np.nanmedian(wolf_values - dog_values)),
                "wolf_minus_dog_seed_range95": np.nanquantile(wolf_values - dog_values,
                                                               [0.025, 0.975]).tolist(),
                "wolf_minus_dog_block_seed_ci95": np.nanquantile(draws, [0.025, 0.975]).tolist(),
                "bootstrap_fraction_le_zero": float(np.nanmean(draws <= 0)),
            }
        output["scales"][str(scale)] = scale_out

    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    with open(args.table, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--wolf", required=True)
    prep.add_argument("--dog", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--replicates", type=int, default=20)
    prep.add_argument("--seed", type=int, default=20260814)
    prep.set_defaults(func=prepare)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    summary.add_argument("--replicates", type=int, default=20)
    summary.add_argument("--bootstrap", type=int, default=10_000)
    summary.add_argument("--seed", type=int, default=20260814)
    summary.add_argument("--output", required=True)
    summary.add_argument("--table", required=True)
    summary.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
