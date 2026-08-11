"""Generate a cross-map-blind, structured-selfing robustness campaign.

The prior deliberately mixes panmictic panels with pooled samples from recently
diverged demes.  Two sampling designs reproduce only the observed Arabis panel
sizes and population counts; no linkage-map value is read or encoded here.
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import get_context
from pathlib import Path

import msprime
import numpy as np

from fastrho.simulate import RecombPriors, make_recombination_map

SEQUENCE_LENGTH = 1_000_000
MAX_RHO_TOTAL = 6_000.0
PANEL_SIZES = (8, 12, 16, 24, 25, 32, 50)
DESIGN_NAMES = (
    "panmictic",
    "diffuse_island",
    "nemorensis_panel",
    "sagittata_panel",
)
DESIGN_WEIGHTS = (0.30, 0.30, 0.20, 0.20)


def draw_selfing(rng: np.random.Generator) -> float:
    """Cover plausible partial-to-near-complete selfing without species labels."""
    component = int(rng.choice(3, p=(0.20, 0.50, 0.30)))
    bounds = ((0.70, 0.90), (0.90, 0.98), (0.98, 0.997))[component]
    return float(rng.uniform(*bounds))


def random_composition(n: int, rng: np.random.Generator) -> list[int]:
    k = int(rng.integers(2, min(5, n) + 1))
    weights = rng.dirichlet(np.full(k, 0.7))
    counts = np.ones(k, dtype=int)
    counts += rng.multinomial(n - k, weights)
    return counts.tolist()


def draw_design(rng: np.random.Generator) -> tuple[str, list[int]]:
    design = str(rng.choice(DESIGN_NAMES, p=DESIGN_WEIGHTS))
    if design == "panmictic":
        return design, [int(rng.choice(PANEL_SIZES))]
    if design == "nemorensis_panel":
        return design, [6, 1, 1, 1, 1, 1, 1]
    if design == "sagittata_panel":
        return design, [15, 5, 5]
    n = int(rng.choice(PANEL_SIZES))
    return design, random_composition(n, rng)


def structured_demography(
    effective_ne: float,
    counts: list[int],
    rng: np.random.Generator,
) -> tuple[msprime.Demography, list[msprime.SampleSet], dict]:
    """Recently diverged demes with post-split gene flow and size changes."""
    demography = msprime.Demography()
    demography.add_population(name="ancestral", initial_size=effective_ne)
    current_sizes = []
    ancient_sizes = []
    population_names = []
    for index in range(len(counts)):
        name = f"deme{index}"
        population_names.append(name)
        current = effective_ne * 10.0 ** rng.uniform(-0.45, 0.45)
        ancient = effective_ne * 10.0 ** rng.uniform(-0.30, 0.30)
        current_sizes.append(float(current))
        ancient_sizes.append(float(ancient))
        demography.add_population(name=name, initial_size=current)

    split_time = effective_ne * 10.0 ** rng.uniform(-1.25, 0.35)
    recent_time = split_time * rng.uniform(0.03, 0.45)
    for name, ancient in zip(population_names, ancient_sizes):
        demography.add_population_parameters_change(
            time=recent_time, initial_size=ancient, population=name
        )

    migration_number = 10.0 ** rng.uniform(-2.0, 1.0)
    migration_rate = migration_number / (4.0 * effective_ne)
    for i, left in enumerate(population_names):
        for right in population_names[i + 1 :]:
            demography.set_symmetric_migration_rate(
                populations=[left, right], rate=migration_rate
            )
    demography.add_population_split(
        time=split_time, derived=population_names, ancestral="ancestral"
    )
    demography.sort_events()
    samples = [
        msprime.SampleSet(num_samples=count, population=name, ploidy=1)
        for name, count in zip(population_names, counts)
    ]
    details = {
        "split_time_generations": float(split_time),
        "recent_size_change_time_generations": float(recent_time),
        "migration_rate": float(migration_rate),
        "four_Ne_m": float(migration_number),
        "current_deme_sizes": current_sizes,
        "pre_recent_change_deme_sizes": ancient_sizes,
    }
    return demography, samples, details


def generate(index: int):
    rng = np.random.default_rng(9_000_000 + index)
    design, counts = draw_design(rng)
    n_haplotypes = int(sum(counts))
    mutation_rate = 10.0 ** rng.uniform(-8.35, -7.95)
    census_ne = 10.0 ** rng.uniform(4.7, 5.7)
    meiotic_rate = 10.0 ** rng.uniform(-8.15, -7.35)
    selfing = draw_selfing(rng)
    inbreeding = selfing / (2.0 - selfing)
    effective_ne = census_ne / (1.0 + inbreeding)
    map_kind = str(rng.choice(("hotspot", "gp", "pericentromere"), p=(0.35, 0.50, 0.15)))
    meiotic_map = make_recombination_map(
        SEQUENCE_LENGTH,
        rng,
        kind=map_kind,
        mean_rate=meiotic_rate,
        priors=RecombPriors(sequence_length=SEQUENCE_LENGTH),
    )
    meiotic_mean = float(
        np.average(np.asarray(meiotic_map.rate, float), weights=np.diff(meiotic_map.position))
    )
    rho_total = 4.0 * effective_ne * meiotic_mean * (1.0 - inbreeding) * SEQUENCE_LENGTH
    ne_cap_applied = rho_total > MAX_RHO_TOTAL
    if ne_cap_applied:
        scale = MAX_RHO_TOTAL / rho_total
        census_ne *= scale
        effective_ne *= scale
    effective_rate = np.asarray(meiotic_map.rate, float) * (1.0 - inbreeding)
    effective_map = msprime.RateMap(
        position=np.asarray(meiotic_map.position, float), rate=effective_rate
    )
    ancestry_seed = int(rng.integers(1, 2**31))
    mutation_seed = int(rng.integers(1, 2**31))

    structure = {
        "split_time_generations": None,
        "recent_size_change_time_generations": None,
        "migration_rate": None,
        "four_Ne_m": None,
        "current_deme_sizes": [float(effective_ne)],
        "pre_recent_change_deme_sizes": [float(effective_ne)],
    }
    if len(counts) == 1:
        ts = msprime.sim_ancestry(
            samples=n_haplotypes,
            ploidy=1,
            population_size=effective_ne,
            recombination_rate=effective_map,
            sequence_length=SEQUENCE_LENGTH,
            random_seed=ancestry_seed,
        )
    else:
        demography, samples, structure = structured_demography(effective_ne, counts, rng)
        ts = msprime.sim_ancestry(
            samples=samples,
            demography=demography,
            recombination_rate=effective_map,
            sequence_length=SEQUENCE_LENGTH,
            random_seed=ancestry_seed,
        )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=mutation_seed)
    meta = {
        "seed": int(index),
        "n_samples": n_haplotypes,
        "n_haplotypes": n_haplotypes,
        "sample_counts_by_deme": counts,
        "n_demes": len(counts),
        "design": design,
        "mutation_rate": float(mutation_rate),
        "meiotic_mean_rate": meiotic_mean,
        "mean_rate": float(
            np.average(effective_rate, weights=np.diff(effective_map.position))
        ),
        "census_Ne": float(census_ne),
        "Ne": float(effective_ne),
        "selfing": float(selfing),
        "inbreeding_coefficient": float(inbreeding),
        "rho_total_cap": float(MAX_RHO_TOTAL),
        "Ne_cap_applied": bool(ne_cap_applied),
        "map_kind": map_kind,
        "sequence_length": float(SEQUENCE_LENGTH),
        "window_size": 2000,
        "num_sites": int(ts.num_sites),
        "campaign": "arabis_structured_selfing_v1",
        "panel_sizes": list(PANEL_SIZES),
        "cross_map_used": False,
        **structure,
    }
    return ts, effective_map, meta


def dump(task: tuple[int, str]) -> str:
    index, output = task
    base = Path(output) / f"ts_{index:08d}"
    if base.with_suffix(".trees").exists() and base.with_suffix(".npz").exists():
        return str(base)
    ts, recombination_map, meta = generate(index)
    temporary = base.with_name(base.name + f".tmp-{os.getpid()}")
    ts.dump(str(temporary) + ".trees")
    np.savez(
        str(temporary) + ".npz",
        map_position=np.asarray(recombination_map.position),
        map_rate=np.asarray(recombination_map.rate),
        meta=json.dumps(meta),
    )
    os.replace(str(temporary) + ".trees", str(base) + ".trees")
    os.replace(str(temporary) + ".npz", str(base) + ".npz")
    return str(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(args.offset + i, str(args.out_dir)) for i in range(args.count)]
    with get_context("fork").Pool(args.workers) as pool:
        for _ in pool.imap_unordered(dump, tasks):
            pass
    print(f"generated={args.count} offset={args.offset} out={args.out_dir}")


if __name__ == "__main__":
    main()
