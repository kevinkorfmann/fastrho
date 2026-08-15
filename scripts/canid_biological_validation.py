"""Biological validation of canid maps using canFam3 GC and promoter density."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


WINDOW = 100_000


def load_gc(path: str, chromosome: str, n_windows: int) -> np.ndarray:
    weighted_sum = np.zeros(n_windows, dtype=float)
    weight = np.zeros(n_windows, dtype=float)
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 13 or fields[0] != chromosome:
                continue
            start, end = int(fields[1]), int(fields[2])
            valid_count, sum_data = float(fields[10]), float(fields[11])
            if valid_count <= 0 or end <= start:
                continue
            mean_gc = sum_data / valid_count
            first = max(0, start // WINDOW)
            last = min(n_windows - 1, (end - 1) // WINDOW)
            for index in range(first, last + 1):
                overlap = max(0, min(end, (index + 1) * WINDOW)
                              - max(start, index * WINDOW))
                if overlap:
                    weighted_sum[index] += mean_gc * overlap
                    weight[index] += overlap
    return np.divide(weighted_sum, weight, out=np.full(n_windows, np.nan), where=weight > 0)


def load_tss(path: str, chromosome: str, n_windows: int) -> tuple[np.ndarray, list[int]]:
    positions = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            fields = line.rstrip().split("\t")
            if len(fields) < 6 or fields[2] != chromosome:
                continue
            start, end = int(fields[4]), int(fields[5])
            tss = start if fields[3] == "+" else end - 1
            if 0 <= tss < n_windows * WINDOW:
                positions.add(tss)
    counts = np.bincount(np.asarray(sorted(positions), dtype=int) // WINDOW,
                         minlength=n_windows).astype(float)
    return counts[:n_windows], sorted(positions)


def correlation(rate, feature, kind):
    valid = np.isfinite(rate) & np.isfinite(feature) & (rate > 0)
    x, y = rate[valid], feature[valid]
    if kind == "spearman":
        return float(spearmanr(x, y).statistic)
    if kind == "log_pearson":
        return float(pearsonr(np.log(x), y).statistic)
    raise ValueError(kind)


def promoter_enrichment(rate, tss):
    valid = np.isfinite(rate) & np.isfinite(tss) & (rate > 0)
    rate, tss = rate[valid], tss[valid]
    low, high = np.quantile(tss, [0.25, 0.75])
    lower = rate[tss <= low]
    upper = rate[tss >= high]
    return float(np.exp(np.mean(np.log(upper)) - np.mean(np.log(lower))))


def partial_coefficients(rate, gc, tss, snp_density):
    valid = (np.isfinite(rate) & np.isfinite(gc) & np.isfinite(tss)
             & np.isfinite(snp_density) & (rate > 0))
    y = np.log(rate[valid])
    x = np.column_stack((gc[valid], np.log1p(tss[valid]),
                         np.log1p(snp_density[valid])))
    y = (y - y.mean()) / y.std()
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    design = np.column_stack((np.ones(len(y)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    return beta[1:]


def resample_blocks(arrays, rng, block_windows=50):
    n = min(map(len, arrays))
    starts = np.arange(0, n, block_windows)
    chosen = rng.choice(starts, len(starts), replace=True)
    output = [[] for _ in arrays]
    for start in chosen:
        stop = min(start + block_windows, n)
        for target, values in zip(output, arrays):
            target.append(np.asarray(values[start:stop]))
    return [np.concatenate(parts) for parts in output]


def summarize_map(rate, gc, tss, snp_density, rng, replicates):
    point = {
        "gc_spearman": correlation(rate, gc, "spearman"),
        "gc_log_pearson": correlation(rate, gc, "log_pearson"),
        "tss_spearman": correlation(rate, tss, "spearman"),
        "tss_log_pearson": correlation(rate, np.log1p(tss), "log_pearson"),
        "promoter_density_q4_vs_q1_rate_ratio": promoter_enrichment(rate, tss),
    }
    beta = partial_coefficients(rate, gc, tss, snp_density)
    point["partial_beta_gc"] = float(beta[0])
    point["partial_beta_log1p_tss"] = float(beta[1])
    point["partial_beta_log1p_snp_density"] = float(beta[2])
    draws = {key: np.empty(replicates) for key in point}
    for index in range(replicates):
        r, g, t, s = resample_blocks((rate, gc, tss, snp_density), rng)
        draws["gc_spearman"][index] = correlation(r, g, "spearman")
        draws["gc_log_pearson"][index] = correlation(r, g, "log_pearson")
        draws["tss_spearman"][index] = correlation(r, t, "spearman")
        draws["tss_log_pearson"][index] = correlation(r, np.log1p(t), "log_pearson")
        draws["promoter_density_q4_vs_q1_rate_ratio"][index] = promoter_enrichment(r, t)
        b = partial_coefficients(r, g, t, s)
        draws["partial_beta_gc"][index] = b[0]
        draws["partial_beta_log1p_tss"][index] = b[1]
        draws["partial_beta_log1p_snp_density"][index] = b[2]
    return {key: {"estimate": value,
                  "ci95_5mb_block_bootstrap": np.nanquantile(draws[key], [0.025, 0.975]).tolist()}
            for key, value in point.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wolf-map", required=True)
    parser.add_argument("--dog-map", required=True)
    parser.add_argument("--wolf-genotypes", required=True)
    parser.add_argument("--dog-genotypes", required=True)
    parser.add_argument("--gc-wig", required=True)
    parser.add_argument("--refseq", required=True)
    parser.add_argument("--chrom", default="chr1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    wolf = np.load(args.wolf_map)
    dog = np.load(args.dog_map)
    n = min(len(wolf["pred"]), len(dog["pred"]), len(wolf["truth"]))
    centers = np.asarray(wolf["centers"][:n], dtype=float)
    gc = load_gc(args.gc_wig, args.chrom, n)
    tss, tss_positions = load_tss(args.refseq, args.chrom, n)
    wolf_genotypes = np.load(args.wolf_genotypes, allow_pickle=True)
    dog_genotypes = np.load(args.dog_genotypes, allow_pickle=True)
    wolf_snp = np.bincount(np.asarray(wolf_genotypes["pos"], dtype=int) // WINDOW,
                           minlength=n)[:n].astype(float)
    dog_snp = np.bincount(np.asarray(dog_genotypes["pos"], dtype=int) // WINDOW,
                          minlength=n)[:n].astype(float)
    maps = {"Campbell pedigree": (np.asarray(wolf["truth"][:n], dtype=float), dog_snp),
            "wolf LD": (np.asarray(wolf["pred"][:n], dtype=float), wolf_snp),
            "dog LD": (np.asarray(dog["pred"][:n], dtype=float), dog_snp)}
    rng = np.random.default_rng(args.seed)
    result = {
        "description": "CanFam3 chr1 biological validation in aligned 100-kb windows.",
        "chromosome": args.chrom, "window_bp": WINDOW, "n_windows": n,
        "n_unique_tss": len(tss_positions), "block_size_bp": 5_000_000,
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
        "sources": {"gc": args.gc_wig, "tss": args.refseq,
                    "wolf_map": args.wolf_map, "dog_map": args.dog_map,
                    "wolf_genotypes": args.wolf_genotypes,
                    "dog_genotypes": args.dog_genotypes},
        "prdm9_analysis": {
            "performed": False,
            "reason": ("Canids lack functional PRDM9; applying a human or mouse PRDM9 motif to "
                       "canFam3 would not be a defensible biological validation."),
            "primary_references": ["Axelsson et al. 2012, Genome Research, doi:10.1101/gr.124123.111",
                                   "Auton et al. 2013, PLoS Genetics, doi:10.1371/journal.pgen.1003984"],
        },
        "maps": {name: summarize_map(rate, gc, tss, density, rng, args.bootstrap)
                 for name, (rate, density) in maps.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with open(args.table, "w", newline="") as handle:
        fields = ["center_mb", "gc_percent", "unique_tss", "wolf_snp_count",
                  "dog_snp_count", "campbell_pedigree", "wolf_ld", "dog_ld"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for index in range(n):
            writer.writerow(dict(center_mb=centers[index], gc_percent=gc[index],
                                 unique_tss=int(tss[index]), wolf_snp_count=wolf_snp[index],
                                 dog_snp_count=dog_snp[index],
                                 campbell_pedigree=maps["Campbell pedigree"][0][index],
                                 wolf_ld=maps["wolf LD"][0][index],
                                 dog_ld=maps["dog LD"][0][index]))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
