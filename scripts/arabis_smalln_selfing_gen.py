"""Generate a panel-size-matched selfing campaign without consulting Arabis maps."""

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
PANEL_SIZES = (8, 12, 16, 24, 25, 32, 50)


def generate(index: int):
    rng = np.random.default_rng(1_000 + index)
    n = int(rng.choice(PANEL_SIZES))
    mutation_rate = 10.0 ** rng.uniform(-8.3, -8.0)
    census_ne = 10.0 ** rng.uniform(5.0, 5.6)
    meiotic_rate = 10.0 ** rng.uniform(-8.0, -7.4)
    selfing = rng.uniform(0.90, 0.99)
    inbreeding = selfing / (2.0 - selfing)
    effective_ne = census_ne / (1.0 + inbreeding)
    kind = "hotspot" if rng.random() < 0.5 else "gp"
    recombination_map = make_recombination_map(
        SEQUENCE_LENGTH,
        rng,
        kind=kind,
        mean_rate=meiotic_rate,
        priors=RecombPriors(sequence_length=SEQUENCE_LENGTH),
    )
    effective_rate = np.asarray(recombination_map.rate, float) * (1.0 - inbreeding)
    effective_map = msprime.RateMap(
        position=np.asarray(recombination_map.position, float), rate=effective_rate
    )
    ancestry_seed = int(rng.integers(1, 2**31))
    mutation_seed = int(rng.integers(1, 2**31))
    ts = msprime.sim_ancestry(
        samples=n,
        ploidy=1,
        population_size=effective_ne,
        recombination_rate=effective_map,
        sequence_length=SEQUENCE_LENGTH,
        random_seed=ancestry_seed,
    )
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=mutation_seed)
    meta = {
        "seed": int(index),
        "n_samples": n,
        "n_haplotypes": n,
        "mutation_rate": float(mutation_rate),
        "mean_rate": float(np.average(effective_rate, weights=np.diff(recombination_map.position))),
        "Ne": float(effective_ne),
        "selfing": float(selfing),
        "sequence_length": float(SEQUENCE_LENGTH),
        "window_size": 2000,
        "num_sites": int(ts.num_sites),
        "campaign": "arabis_smalln_self3",
        "panel_sizes": list(PANEL_SIZES),
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
