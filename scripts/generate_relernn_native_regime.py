#!/usr/bin/env python3
"""Generate the frozen constant-within-region ReLERNN intended-regime suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Optional, Tuple

import msprime
import numpy as np


SCENARIO_INDEX = {
    "constant": 0,
    "bottleneck": 1,
    "expansion": 2,
    "decode": 3,
    "hapmap": 4,
    "dog": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_map_rate(path: Path) -> Tuple[float, float]:
    with np.load(path, allow_pickle=True) as archive:
        position = np.asarray(archive["map_position"], dtype=float)
        rate = np.asarray(archive["map_rate"], dtype=float)
    segment = np.diff(position)
    keep = np.isfinite(rate) & (segment > 0)
    if position.ndim != 1 or rate.ndim != 1 or rate.size + 1 != position.size:
        raise ValueError(f"Malformed rate map: {path}")
    if not np.any(keep):
        raise ValueError(f"Rate map has no finite positive-length segments: {path}")
    length = float(segment[keep].sum())
    return float(np.average(rate[keep], weights=segment[keep])), length


def demography(
    name: str, ne: float
) -> Tuple[Optional[msprime.Demography], Optional[float]]:
    if name in {"constant", "realmap"}:
        return None, ne
    model = msprime.Demography()
    model.add_population(initial_size=ne)
    if name == "bottleneck":
        model.add_population_parameters_change(time=1000, initial_size=ne / 10)
        model.add_population_parameters_change(time=3000, initial_size=ne)
    elif name == "expansion":
        model.add_population_parameters_change(time=2000, initial_size=ne / 10)
    else:
        raise ValueError(f"Unknown demography: {name}")
    return model, None


def valid_biallelic_columns(genotypes: np.ndarray) -> np.ndarray:
    return (genotypes.min(axis=0) == 0) & (genotypes.max(axis=0) == 1)


def write_vcf(
    path: Path,
    chrom: str,
    genotypes: np.ndarray,
    positions: np.ndarray,
    sequence_length: int,
) -> int:
    keep = valid_biallelic_columns(genotypes)
    genotypes = genotypes[:, keep]
    positions = positions[keep]
    last = -1
    with path.open("w") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(f"##contig=<ID={chrom},length={sequence_length}>\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(f"tsk_{index}" for index in range(genotypes.shape[0] // 2))
            + "\n"
        )
        for column in range(genotypes.shape[1]):
            position = int(positions[column])
            if position <= last:
                position = last + 1
            if position > sequence_length:
                raise ValueError(f"VCF coordinate exceeds sequence length: {position}")
            last = position
            haplotypes = genotypes[:, column]
            calls = "\t".join(
                f"{haplotypes[2 * sample]}|{haplotypes[2 * sample + 1]}"
                for sample in range(genotypes.shape[0] // 2)
            )
            handle.write(
                f"{chrom}\t{position}\t.\tA\tT\t.\tPASS\t.\tGT\t{calls}\n"
            )
    return int(keep.sum())


def artifact(stem: Path, suffix: str) -> Path:
    """Append an extension without treating the region number as a suffix."""

    return Path(f"{stem}{suffix}")


def generate(source_dir: Path, output_dir: Path, seed_base: int) -> dict:
    config = json.loads((source_dir / "config.json").read_text())
    scenario = str(config["name"]).lower()
    scenario = {"decode": "decode", "hapmap": "hapmap"}.get(scenario, scenario)
    if scenario not in SCENARIO_INDEX:
        raise ValueError(f"Unrecognized scenario name in config: {config['name']}")
    unexpected = list(output_dir.iterdir()) if output_dir.exists() else []
    if unexpected:
        raise FileExistsError(f"Refusing to reuse nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_truth = sorted(source_dir.glob("region_*.npz"))
    if len(source_truth) != int(config["n_regions"]):
        raise ValueError(
            f"Expected {config['n_regions']} source regions, found {len(source_truth)}"
        )
    sequence_length = int(config["seq_len"])
    n_diploid = int(config["n_dip"])
    mutation_rate = float(config["mu"])
    ne = float(config["Ne"])
    demographic_name = str(config["demography"])
    model, population_size = demography(demographic_name, ne)

    output_config = dict(config)
    output_config.update(
        {
            "name": scenario,
            "native_regime": True,
            "native_regime_map_rule": "constant at frozen source-region length-weighted mean",
            "native_regime_source": str(source_dir.resolve()),
            "native_regime_seed_base": seed_base,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(output_config, indent=2, sort_keys=True) + "\n"
    )

    provenance = {
        "scenario": scenario,
        "source_directory": str(source_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "sequence_length_bp": sequence_length,
        "n_diploid": n_diploid,
        "mutation_rate": mutation_rate,
        "Ne": ne,
        "demography": demographic_name,
        "seed_base": seed_base,
        "regions": [],
    }
    for region_index, source_path in enumerate(source_truth):
        match = re.fullmatch(r"region_(\d{3})\.npz", source_path.name)
        if match is None or int(match.group(1)) != region_index:
            raise ValueError(f"Unexpected source region order: {source_path}")
        rate, source_length = mean_map_rate(source_path)
        if not np.isclose(source_length, sequence_length):
            raise ValueError(
                f"Source map length mismatch for {source_path}: {source_length}"
            )
        rate_map = msprime.RateMap(
            position=np.array([0.0, float(sequence_length)]),
            rate=np.array([rate]),
        )
        ancestry_seed = seed_base + SCENARIO_INDEX[scenario] * 1000 + 2 * region_index
        mutation_seed = ancestry_seed + 1
        ancestry = {
            "samples": n_diploid,
            "recombination_rate": rate_map,
            "sequence_length": sequence_length,
            "random_seed": ancestry_seed,
        }
        if model is None:
            ancestry["population_size"] = population_size
        else:
            ancestry["demography"] = model
        tree_sequence = msprime.sim_ancestry(**ancestry)
        tree_sequence = msprime.sim_mutations(
            tree_sequence,
            rate=mutation_rate,
            random_seed=mutation_seed,
        )
        stem = output_dir / f"region_{region_index:03d}"
        tree_sequence.dump(str(artifact(stem, ".trees")))
        metadata = {
            "Ne": ne,
            "mutation_rate": mutation_rate,
            "n_samples": n_diploid,
            "sequence_length": sequence_length,
            "window_size": 2000,
            "contig": f"chr{region_index + 1}",
            "native_regime": True,
            "source_region_mean_rate": rate,
        }
        np.savez(
            artifact(stem, ".npz"),
            map_position=np.array([0.0, float(sequence_length)]),
            map_rate=np.array([rate]),
            meta=json.dumps(metadata),
        )
        genotypes = tree_sequence.genotype_matrix().T.astype(np.int8)
        positions = tree_sequence.tables.sites.position
        segregating = write_vcf(
            artifact(stem, ".vcf"),
            f"chr{region_index + 1}",
            genotypes,
            positions,
            sequence_length,
        )
        provenance["regions"].append(
            {
                "region": region_index,
                "source_truth": str(source_path.resolve()),
                "source_truth_sha256": sha256(source_path),
                "constant_rate_per_bp": rate,
                "ancestry_seed": ancestry_seed,
                "mutation_seed": mutation_seed,
                "tree_sequence_sites": int(tree_sequence.num_sites),
                "biallelic_segregating_sites": segregating,
                "truth_sha256": sha256(artifact(stem, ".npz")),
                "vcf_sha256": sha256(artifact(stem, ".vcf")),
                "trees_sha256": sha256(artifact(stem, ".trees")),
            }
        )
        print(
            f"{scenario} region {region_index:03d}: rate={rate:.8g}, "
            f"sites={tree_sequence.num_sites}, biallelic={segregating}",
            flush=True,
        )
    with (output_dir / "genome.bed").open("w") as handle:
        for region_index in range(len(source_truth)):
            handle.write(f"chr{region_index + 1}\t0\t{sequence_length}\n")
    provenance["config_sha256"] = sha256(output_dir / "config.json")
    provenance["genome_bed_sha256"] = sha256(output_dir / "genome.bed")
    (output_dir / "native_regime_generation.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed-base", type=int, default=92001)
    args = parser.parse_args()
    provenance = generate(
        args.source_dir.resolve(), args.output_dir.resolve(), args.seed_base
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
