"""Generate small-panel wolf simulations with panmictic and structured histories.

This is a targeted follow-up to the globally pooled Plassais wolf analysis. It expands the
canid training prior to 8--20 diploids and includes two- and three-deme island histories, while
retaining the canid mutation-rate and recombination-map distributions used by ``dog_gen.py``.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import get_context

import msprime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.simulate import RecombPriors, make_recombination_map


LENGTH = 1_000_000
MAX_RHO = 5_000.0


def _sample_counts(total: int, demes: int, rng: np.random.Generator) -> list[int]:
    counts = np.ones(demes, dtype=int)
    for index in rng.integers(0, demes, total - demes):
        counts[index] += 1
    return counts.tolist()


def generate(seed: int):
    rng = np.random.default_rng(seed)
    n_diploid = int(rng.choice([8, 10, 12, 16, 18, 20]))
    mutation_rate = 10 ** rng.uniform(np.log10(2e-9), np.log10(6e-9))
    mean_rate = 10 ** rng.uniform(np.log10(5e-9), np.log10(5e-8))
    ne = 10 ** rng.uniform(np.log10(1.5e4), np.log10(8.0e4))
    if 4 * ne * mean_rate * LENGTH > MAX_RHO:
        mean_rate *= MAX_RHO / (4 * ne * mean_rate * LENGTH)

    map_kind = "hotspot" if rng.random() < 0.5 else "gp"
    rate_map = make_recombination_map(
        LENGTH,
        rng,
        kind=map_kind,
        mean_rate=mean_rate,
        priors=RecombPriors(sequence_length=LENGTH),
    )

    structured = rng.random() < 0.55
    if structured:
        demes = int(rng.choice([2, 3]))
        migration = 10 ** rng.uniform(-5.2, -3.0)
        demography = msprime.Demography.island_model(
            initial_size=[ne] * demes,
            migration_rate=migration,
        )
        counts = _sample_counts(n_diploid, demes, rng)
        samples = {f"pop_{index}": count for index, count in enumerate(counts)}
        mode = f"island_{demes}deme"
    else:
        demes = 1
        migration = 0.0
        demography = msprime.Demography.isolated_model([ne])
        counts = [n_diploid]
        samples = {"pop_0": n_diploid}
        mode = "panmictic"

    ancestry = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        recombination_rate=rate_map,
        sequence_length=LENGTH,
        random_seed=int(rng.integers(1, 2**31)),
    )
    sequence = msprime.sim_mutations(
        ancestry,
        rate=mutation_rate,
        random_seed=int(rng.integers(1, 2**31)),
    )
    metadata = {
        "n_samples": n_diploid,
        "mutation_rate": float(mutation_rate),
        "mean_rate": float(np.average(rate_map.rate, weights=np.diff(rate_map.position))),
        "Ne": float(ne),
        "Ne_present": float(ne),
        "Ne_anc": float(ne),
        "demography": f"wolf_{mode}",
        "mode": mode,
        "n_demes": demes,
        "migration_rate": float(migration),
        "sample_counts": counts,
        "sequence_length": float(LENGTH),
        "window_size": 2_000,
        "num_sites": int(sequence.num_sites),
    }
    return sequence, rate_map, metadata


def dump(index: int, output: str) -> None:
    stem = os.path.join(output, f"ts_{index:08d}")
    if os.path.exists(stem + ".trees"):
        return
    sequence, rate_map, metadata = generate(8_000_000 + index)
    sequence.dump(stem + ".trees")
    np.savez(
        stem + ".npz",
        map_position=np.asarray(rate_map.position),
        map_rate=np.asarray(rate_map.rate),
        meta=json.dumps(metadata),
    )


def main() -> None:
    output = sys.argv[1]
    count = int(sys.argv[2])
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    os.makedirs(output, exist_ok=True)
    with get_context("fork").Pool(workers) as pool:
        pool.starmap(dump, [(offset + index, output) for index in range(count)])
    print(f"generated {count} wolf simulations in {output}")


if __name__ == "__main__":
    main()
