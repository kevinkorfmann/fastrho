"""Diagnose information and mating-system differences in the Arabis WGS panels.

This script intentionally uses no linkage-map information.  It reports raw-call
heterozygosity/F_IS proxies and properties of the homozygous complete-case VCFs
used by fastrho, including the minor-allele-count spectrum and effective rank.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_gt(value: str, gt_index: int, dp_index: int) -> tuple[int, int, int] | None:
    fields = value.split(":")
    if gt_index >= len(fields) or dp_index >= len(fields):
        return None
    gt = fields[gt_index].replace("|", "/").split("/")
    try:
        dp = int(fields[dp_index])
    except (ValueError, TypeError):
        return None
    if len(gt) != 2 or "." in gt or dp < 5 or dp > 60:
        return None
    try:
        a, b = int(gt[0]), int(gt[1])
    except ValueError:
        return None
    if a not in (0, 1) or b not in (0, 1):
        return None
    return a, b, dp


def load_sheet(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    species, populations = {}, {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            species[row["accession"]] = row["species"]
            populations[row["accession"]] = row["real_population"]
    return species, populations


def raw_call_metrics(vcf: Path, sample_species: dict[str, str]) -> dict:
    samples: list[str] = []
    callable_n: Counter[str] = Counter()
    het_n: Counter[str] = Counter()
    species_obs = Counter()
    species_exp = Counter()
    species_sites = Counter()
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip().split("\t")[9:]
                continue
            fields = line.rstrip().split("\t")
            if len(fields[3]) != 1 or len(fields[4]) != 1:
                continue
            fmt = fields[8].split(":")
            if "GT" not in fmt or "DP" not in fmt:
                continue
            gi, di = fmt.index("GT"), fmt.index("DP")
            calls: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for sample, value in zip(samples, fields[9:]):
                parsed = parse_gt(value, gi, di)
                if parsed is None:
                    continue
                a, b, _ = parsed
                callable_n[sample] += 1
                het_n[sample] += a != b
                calls[sample_species[sample]].append((a, b))
            for species, genotypes in calls.items():
                expected_n = sum(sample_species[s] == species for s in samples)
                if len(genotypes) != expected_n:
                    continue
                alt = sum(a + b for a, b in genotypes)
                if alt in (0, 2 * len(genotypes)):
                    continue
                p = alt / (2 * len(genotypes))
                species_obs[species] += sum(a != b for a, b in genotypes)
                species_exp[species] += len(genotypes) * 2 * p * (1 - p)
                species_sites[species] += 1
    per_sample = {}
    for sample in samples:
        per_sample[sample] = {
            "species": sample_species[sample],
            "callable_variant_sites": callable_n[sample],
            "heterozygous_calls": het_n[sample],
            "heterozygosity_per_callable_variant": het_n[sample] / max(1, callable_n[sample]),
        }
    by_species = {}
    for species in sorted(set(sample_species.values())):
        f_is = 1.0 - species_obs[species] / species_exp[species] if species_exp[species] else None
        selfing = 2 * f_is / (1 + f_is) if f_is is not None and f_is > -1 else None
        vals = [v["heterozygosity_per_callable_variant"] for v in per_sample.values()
                if v["species"] == species]
        by_species[species] = {
            "complete_polymorphic_sites_for_fis": species_sites[species],
            "observed_heterozygotes": species_obs[species],
            "expected_heterozygotes_under_hwe": species_exp[species],
            "F_IS_proxy": f_is,
            "selfing_proxy_2F_over_1plusF": selfing,
            "median_per_sample_heterozygosity": float(np.median(vals)),
            "range_per_sample_heterozygosity": [float(np.min(vals)), float(np.max(vals))],
        }
    return {"per_sample": per_sample, "by_species": by_species}


def filtered_metrics(vcf: Path, populations: dict[str, str], max_matrix_sites: int = 100_000) -> dict:
    samples: list[str] = []
    mac = Counter()
    chrom_sites = Counter()
    matrices: list[np.ndarray] = []
    positions: list[tuple[str, int]] = []
    total = 0
    with gzip.open(vcf, "rt") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip().split("\t")[9:]
                continue
            fields = line.rstrip().split("\t")
            fmt = fields[8].split(":")
            gi = fmt.index("GT")
            alleles = []
            for value in fields[9:]:
                gt = value.split(":")[gi].replace("|", "/").split("/")
                if len(gt) != 2 or gt[0] != gt[1] or gt[0] not in ("0", "1"):
                    raise ValueError(f"non-homozygous/complete genotype in {vcf}")
                alleles.append(int(gt[0]))
            arr = np.asarray(alleles, dtype=np.int8)
            count = int(min(arr.sum(), len(arr) - arr.sum()))
            mac[count] += 1
            chrom_sites[fields[0]] += 1
            total += 1
            # Deterministic thinning keeps matrix work bounded and genome-wide.
            if len(matrices) < max_matrix_sites:
                matrices.append(arr)
                positions.append((fields[0], int(fields[1])))
            elif total % max(2, total // max_matrix_sites) == 0:
                j = (total * 2654435761) % max_matrix_sites
                matrices[j] = arr
                positions[j] = (fields[0], int(fields[1]))
    x = np.stack(matrices).T.astype(float)
    pairwise = {}
    for i, a in enumerate(samples):
        for j in range(i + 1, len(samples)):
            pairwise[f"{a}|{samples[j]}"] = float(np.mean(x[i] != x[j]))
    within, between = [], []
    for key, value in pairwise.items():
        a, b = key.split("|")
        (within if populations[a] == populations[b] else between).append(value)
    p = x.mean(axis=0)
    variable = (p > 0) & (p < 1)
    z = (x[:, variable] - p[variable]) / np.sqrt(p[variable] * (1 - p[variable]))
    gram = z @ z.T / max(1, z.shape[1])
    eig = np.linalg.eigvalsh(gram)
    eig = np.maximum(eig, 0)
    effective_rank = float(eig.sum() ** 2 / np.square(eig).sum()) if np.square(eig).sum() else 0.0
    singleton_fraction = mac[1] / total
    return {
        "n_accessions": len(samples),
        "accessions": samples,
        "n_sites": total,
        "sites_per_chromosome": dict(sorted(chrom_sites.items())),
        "minor_allele_count_spectrum": {str(k): mac[k] for k in sorted(mac)},
        "singleton_site_fraction": singleton_fraction,
        "matrix_sites": x.shape[1],
        "standardized_genotype_effective_rank": effective_rank,
        "leading_genotype_gram_eigenvalues": [float(v) for v in eig[::-1][:5]],
        "pairwise_difference": {
            "within_population_median": float(np.median(within)) if within else None,
            "between_population_median": float(np.median(between)) if between else None,
            "within_population_n_pairs": len(within),
            "between_population_n_pairs": len(between),
            "all_pairs": pairwise,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--sample-sheet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    species, populations = load_sheet(args.sample_sheet)
    result = {
        "interpretation_warning": (
            "F_IS/selfing estimates are descriptive proxies: population structure and variant "
            "ascertainment can inflate them. They are not direct mating-system estimates."
        ),
        "raw_calls": raw_call_metrics(args.workdir / "vcf/all.genome.vcf.gz", species),
        "filtered_panels": {
            name: filtered_metrics(args.workdir / f"vcf/{name}.selfer.vcf.gz", populations)
            for name in ("nemorensis", "sagittata")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            "n": metrics["n_accessions"], "sites": metrics["n_sites"],
            "singleton_fraction": metrics["singleton_site_fraction"],
            "effective_rank": metrics["standardized_genotype_effective_rank"],
        } for name, metrics in result["filtered_panels"].items()
    }, indent=2))
    print(json.dumps(result["raw_calls"]["by_species"], indent=2))


if __name__ == "__main__":
    main()
